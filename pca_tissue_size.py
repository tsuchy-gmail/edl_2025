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
import cv2

test_cases = ["JMR0077", "JMR0299", "JMR2518", "JMR0020", "JMR0465", "JMR2499"]
BATCH_SIZE = 256
DEVICE     = "cuda:1"
MODEL_NAME = "hf-hub:paige-ai/Virchow"

def load_model(name: str, device: str):
    model = timm.create_model(MODEL_NAME, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

def embed(x):
    y = vir(x)
    feats = torch.cat([y[:,0], y[:,1:].mean(1)], -1)
    return feats

def tissue_region_coords_16x(slide, org_patch_size):
    level = 2
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
    print(case)
    wsi_path   = f"/Raw/Kurume_Dataset/JMR_svs/{case}.svs"
    out_path   = f"figure/pca/tissue_area/{case}.png"
    slide   = openslide.OpenSlide(wsi_path)
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

    embeds = np.empty((n_total, 2560), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_total, BATCH_SIZE):
            print(f"{i}/{n_total}")
            batch_coords = coords[i:i+BATCH_SIZE]
            imgs = []
            for x, y in batch_coords:
                img = slide.read_region((x, y), 0, (org_patch_size, org_patch_size)).convert("RGB")
                imgs.append(tfm(img))
            x_tensor = torch.stack(imgs).to(DEVICE)
            out = model(x_tensor).cpu()
            feats = torch.cat([out[:,0], out[:,1:].mean(1)], -1).numpy()
            embeds[i:i+len(batch_coords)] = feats

    #np.savez(f"{case}_embeds.npz", embeds=embeds, coords=coords)
    #print(f"✅ Saved: {case}_embeds.npz")

    pca = PCA(n_components=3, svd_solver="randomized")
    rgb = pca.fit_transform(embeds)         # (N, 3)
    rgb -= rgb.min(axis=0, keepdims=True)
    rgb /= rgb.max(axis=0, keepdims=True) + 1e-7
    rgb = (rgb * 255).astype(np.uint8)

    w0, h0  = slide.level_dimensions[0]
    nx = w0 // org_patch_size
    ny = h0 // org_patch_size
    canvas = np.full((ny*org_patch_size, nx*org_patch_size, 3), 250, dtype=np.uint8)
    for (x, y), color in zip(coords, rgb):
        canvas[y:y+org_patch_size, x:x+org_patch_size] = color

    Image.fromarray(canvas).save(out_path)
    print("saved:", out_path)

if __name__ == "__main__":
    org_patch_size = 64
    case = "JMR2499"
    #for case in test_cases:
    main(case, org_patch_size)

