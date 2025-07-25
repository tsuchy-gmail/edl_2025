import os
import pathlib, math, sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import openslide
from sklearn.decomposition import PCA, IncrementalPCA
import timm
from timm.layers import SwiGLUPacked
import cv2

CASE = "JMR0020"
WSI_PATH   = f"/Raw/Kurume_Dataset/JMR_svs/{CASE}.svs"
OUT_PATH   = f"figure/pca/tissue_area/patch14/{CASE}.png"
PATCH_SIZE = 224
TOKEN_PATCH_SIZE = 14
BATCH_SIZE = 256
DEVICE     = "cuda:1"
MODEL_NAME = "hf-hub:paige-ai/Virchow"

def load_model(name: str, device: str):
    model = timm.create_model(MODEL_NAME, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

def tissue_region_coords_16x(slide):
    level = 2
    org_size = 224
    div_scales = [1, 4, 16]
    div_scale = div_scales[level]
    ratio = 0.5

    start_tup = (0, 0)
    end_tup = slide.level_dimensions[level]

    wsi_lev2 = slide.read_region(start_tup, level, end_tup)
    wsi_lev2 = wsi_lev2.convert("RGB")
    #wsi_lev2.save(f"figure/{case}_lev2.png")
    wsi_np = np.array(wsi_lev2)
    wsi_gray = cv2.cvtColor(wsi_np, cv2.COLOR_RGB2GRAY)
    th, mask = cv2.threshold(wsi_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    print("optimal threshold", th)

    patch_size = org_size // div_scale
    coords = []
    height, width = mask.shape
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = mask[y:y+patch_size, x:x+patch_size]
            if np.mean(patch == 255) >= ratio:
                coords.append((x, y))

    return coords

def pca(embeds, coords, slide):
    print("2")
    flat_embeds = embeds.reshape(-1, 1280)
    print(flat_embeds.shape)

    tmp_batch_size = 8192
    pca = IncrementalPCA(n_components=3, batch_size=tmp_batch_size)
    for i in range(0, flat_embeds.shape[0], tmp_batch_size):
        print("i:",i)
        pca.partial_fit(flat_embeds[i:i+tmp_batch_size])

    print("transform")
    #rgb = pca.transform(flat_embeds)
    print("3")
    #rgb = pca.fit_transform(flat_embeds)         # (N, 3)
    rgb = np.empty((flat_embeds.shape[0], 3), dtype=np.float32)
    for start in range(0, flat_embeds.shape[0], tmp_batch_size):
        print("start:", start)
        rgb[start:start + tmp_batch_size] = pca.transform(flat_embeds[start:start + tmp_batch_size])
    print("4")
    rgb -= rgb.min(axis=0, keepdims=True)
    print("5")
    rgb /= rgb.max(axis=0, keepdims=True) + 1e-7
    print("6")
    rgb = (rgb * 255).astype(np.uint8)
    print("7")

    w0, h0  = slide.level_dimensions[0]
    nx = w0 // PATCH_SIZE
    ny = h0 // PATCH_SIZE
    n_rep = PATCH_SIZE // TOKEN_PATCH_SIZE
    #n_minipatch = n_rep * TOKEN_PATCH_SIZE
    canvas = np.full((ny*PATCH_SIZE, nx*PATCH_SIZE, 3), 250, dtype=np.uint8)
    #for (x, y), color in zip(coords, rgb):
    print("8")
    for idx, (x, y) in enumerate(coords):
        print(f"{idx}/{len(coords)}")
        for i in range(n_rep):
            for j in range(n_rep):
                inner_y = y + i * TOKEN_PATCH_SIZE
                inner_x = x + j * TOKEN_PATCH_SIZE
                canvas[inner_y:inner_y+TOKEN_PATCH_SIZE, inner_x:inner_x+TOKEN_PATCH_SIZE] = rgb[(256 * idx) + (n_rep * i + j)]

    Image.fromarray(canvas).save(OUT_PATH)
    print("saved:", OUT_PATH)

def get_embeds(slide):
    saved_coords_path = f"coords_{CASE}.npy"

    coords_16x = tissue_region_coords_16x(slide)
    coords = np.array(coords_16x) * 16

    n_total = len(coords_16x)
    print("n_patch", n_total)

    model = load_model(MODEL_NAME, DEVICE)
    cfg = model.pretrained_cfg
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    embeds = np.empty((n_total, 256, 1280), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_total, BATCH_SIZE):
            print(f"{i}/{n_total}")
            batch_coords = coords[i:i+BATCH_SIZE]
            imgs = []
            for x, y in batch_coords:
                img = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                imgs.append(tfm(img))
            x_tensor = torch.stack(imgs).to(DEVICE)
            out = model(x_tensor).cpu()
            feats = out[:,1:].numpy()
            print(feats.shape)
            embeds[i:i+len(batch_coords)] = feats

    #np.save(f"saved_embeds/tissue_area/{CASE}_embeds.npy", embeds)
    #np.savez(f"saved_embeds/tissue_area/{CASE}_embeds.npz", embeds=embeds, coords=coords)
    #print(f"✅ Saved: {CASE}_embeds.npz")
    print("fin embeds")

    return embeds, coords

def main():
    print(CASE)
    embeds_save_path = f"saved_embeds/tissue_area/{CASE}_embeds.npz"
    slide = openslide.OpenSlide(WSI_PATH)
    if os.path.exists(embeds_save_path):
        print("exist")
        data = np.load(embeds_save_path)
        embeds, coords = data["embeds"], data["coords"]
    else:
        print("not exist")
        embeds, coords = get_embeds(slide)

    print("start pca")
    pca(embeds, coords, slide)


if __name__ == "__main__":
    main()

