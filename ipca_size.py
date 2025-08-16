import os
import pathlib, math, sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import openslide
from sklearn.decomposition import IncrementalPCA
import timm
from timm.layers import SwiGLUPacked
import cv2

test_cases = ["JMR0077", "JMR0299", "JMR2518", "JMR0020", "JMR0465", "JMR2499"]
BATCH_SIZE = 256
DEVICE     = "cuda:1"
MODEL_NAME = "hf-hub:paige-ai/Virchow"

def load_model(name: str, device: str):
    model = timm.create_model(name, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

def tissue_region_coords_16x(slide, org_patch_size):
    level = 2
    div_scales = [1, 4, 16]
    div_scale = div_scales[level]
    ratio = 0.5

    start_tup = (0, 0)
    end_tup = slide.level_dimensions[level]

    wsi_lev2 = slide.read_region(start_tup, level, end_tup).convert("RGB")
    wsi_np = np.array(wsi_lev2)
    wsi_gray = cv2.cvtColor(wsi_np, cv2.COLOR_RGB2GRAY)
    th, mask = cv2.threshold(wsi_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    print("optimal threshold", th)

    patch_size = org_patch_size // div_scale
    coords = []
    height, width = mask.shape
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = mask[y:y+patch_size, x:x+patch_size]
            if np.mean(patch == 255) >= ratio:
                coords.append((x, y))
    return coords

def main(case, org_patch_size):
    print(f"Processing case: {case}")
    wsi_path = f"/Raw/Kurume_Dataset/JMR_svs/{case}.svs"
    out_path = f"figure/pca/tissue_area/{case}.png"
    slide = openslide.OpenSlide(wsi_path)
    coords_16x = tissue_region_coords_16x(slide, org_patch_size)
    coords = np.array(coords_16x) * 16
    n_total = len(coords_16x)
    print("n_patch", n_total)

    model = load_model(MODEL_NAME, DEVICE)
    cfg = model.pretrained_cfg
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    # 1) IncrementalPCA を初期化
    ipca = IncrementalPCA(n_components=3)

    # 2) 第1パス：partial_fit
    print("Fitting IncrementalPCA...")
    with torch.no_grad():
        for i in range(0, n_total, BATCH_SIZE):
            batch = coords[i:i+BATCH_SIZE]
            imgs = []
            for x, y in batch:
                img = slide.read_region((x, y), 0, (org_patch_size, org_patch_size)).convert("RGB")
                imgs.append(tfm(img))
            x_tensor = torch.stack(imgs).to(DEVICE)
            out = model(x_tensor).cpu()
            feats = torch.cat([out[:,0], out[:,1:].mean(1)], dim=1).numpy()
            ipca.partial_fit(feats)

    # 3) 第2パス：transform と min/max の集計
    print("Transforming and collecting min/max...")
    rgb_min = np.full(3, np.inf)
    rgb_max = np.full(3, -np.inf)
    rgb_batches = []
    with torch.no_grad():
        for i in range(0, n_total, BATCH_SIZE):
            print(f"{i}/{n_total}")
            batch = coords[i:i+BATCH_SIZE]
            imgs = [tfm(slide.read_region((x, y), 0, (org_patch_size, org_patch_size)).convert("RGB"))
                    for x, y in batch]
            x_tensor = torch.stack(imgs).to(DEVICE)
            out = model(x_tensor).cpu()
            feats = torch.cat([out[:,0], out[:,1:].mean(1)], dim=1).numpy()
            rgb_batch = ipca.transform(feats)
            rgb_min = np.minimum(rgb_min, rgb_batch.min(axis=0))
            rgb_max = np.maximum(rgb_max, rgb_batch.max(axis=0))
            rgb_batches.append(rgb_batch)

    # 4) 正規化して uint8 に変換
    print("Normalizing colors...")
    rgb_list = []
    for rgb_batch in rgb_batches:
        norm = (rgb_batch - rgb_min) / (rgb_max - rgb_min + 1e-7)
        rgb_list.append((norm * 255).astype(np.uint8))
    rgb = np.vstack(rgb_list)

    # 5) キャンバス生成と描画
    w0, h0 = slide.level_dimensions[0]
    nx = w0 // org_patch_size
    ny = h0 // org_patch_size
    canvas = np.full((ny*org_patch_size, nx*org_patch_size, 3), 250, dtype=np.uint8)
    for (x, y), color in zip(coords, rgb):
        canvas[y:y+org_patch_size, x:x+org_patch_size] = color

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    org_patch_size = 32
    # 複数ケースを処理する場合は以下をアンコメント
    # for case in test_cases:
    #     main(case, org_patch_size)
    main("JMR2499", org_patch_size)

