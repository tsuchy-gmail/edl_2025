import os
import pathlib, math, sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import openslide
from sklearn.decomposition import PCA
import timm
from timm.layers import SwiGLUPacked

# ---------- 設定 ----------
CASE = "JMR0299"
WSI_PATH   = f"/Raw/Kurume_Dataset/JMR_svs/{CASE}.svs"          # 入力 Whole-Slide Image
OUT_PATH   = f"figure/pca_{CASE}.png"      # 出力可視化画像
PATCH      = 224                  # パッチ幅・高さ（モデル入力サイズ）
BATCH_SIZE = 256                   # GPU メモリに合わせて調整
DEVICE     = "cuda:3"
MODEL_NAME = "hf-hub:paige-ai/Virchow"  # 2 560-d を返すネット
# --------------------------------

def load_model(name: str, device: str):
    model = timm.create_model(MODEL_NAME, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

def get_patches(slide, level0_wh, patch):
    """パッチ左上座標を level0 基準で列挙（端の余りは切り捨て）"""
    w0, h0 = level0_wh
    nx = w0 // patch
    ny = h0 // patch
    for iy in range(ny):
        for ix in range(nx):
            yield ix * patch, iy * patch

def embed(x):                                         # x: (B,3,224,224)
    y = vir(x)                                        # (B,257,1280)
    feats = torch.cat([y[:,0], y[:,1:].mean(1)], -1)  # (B,2560)
    return feats                                      # on GPU
def main():
    saved_coords_path = f"coords_{CASE}.npy"

    if os.path.exists(saved_coords_path):
        coords = np.load(saved_coords_path)
    else:
        slide   = openslide.OpenSlide(WSI_PATH)
        w0, h0  = slide.level_dimensions[0]
        coords  = list(get_patches(slide, (w0, h0), PATCH))
        np.save(f"coords_{CASE}.npy", np.array(coords, dtype=np.int32))

    n_total = len(coords)
    print(f"WSI size: {w0}×{h0} px, patches: {n_total}")

    model = load_model(MODEL_NAME, DEVICE)
    cfg = model.pretrained_cfg
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])



    embeds = np.empty((n_total, 2560), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_total, BATCH_SIZE):
            print(f"{i}/{n_total}")
            batch_coords = coords[i:i+BATCH_SIZE]
            imgs = []
            for x, y in batch_coords:
                # openslide.read_region returns RGBA; .convert("RGB") で 3ch
                img = slide.read_region((x, y), 0, (PATCH, PATCH)).convert("RGB")
                imgs.append(tfm(img))
            x_tensor = torch.stack(imgs).to(DEVICE)
            out = model(x_tensor).cpu()
            feats = torch.cat([out[:,0], out[:,1:].mean(1)], -1).numpy()
            embeds[i:i+len(batch_coords)] = feats

    np.save(f"{CASE}_embeds.npy", embeds)
    print(f"✅ Saved: {CASE}_embeds.npy")

    # ---------- PCA → 0–255 正規化 ----------
    pca = PCA(n_components=3, svd_solver="randomized")
    rgb = pca.fit_transform(embeds)         # (N, 3)
    rgb -= rgb.min(axis=0, keepdims=True)
    rgb /= rgb.max(axis=0, keepdims=True) + 1e-7
    rgb = (rgb * 255).astype(np.uint8)

    # ---------- モザイク画像を作る ----------
    nx = w0 // PATCH
    ny = h0 // PATCH
    canvas = np.zeros((ny*PATCH, nx*PATCH, 3), dtype=np.uint8)
    for (x, y), color in zip(coords, rgb):
        canvas[y:y+PATCH, x:x+PATCH] = color

    Image.fromarray(canvas).save(OUT_PATH)
    print("saved:", OUT_PATH)

if __name__ == "__main__":
    main()

