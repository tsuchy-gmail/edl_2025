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

#CASE = "JMR0299"
CASE = "JMR0465"
WSI_PATH   = f"/Raw/Kurume_Dataset/JMR_svs/{CASE}.svs"
#OUT_PATH   = f"figure/pca/tissue_area/{CASE}.png"
OUT_PATH   = f"{CASE}_test_canvas_16x.png"
PATCH_SIZE = 224
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

def draw_from_embeds(embeds):
    slide = openslide.OpenSlide(WSI_PATH)
    coords= tissue_region_coords_16x(slide)

    #pca = PCA(n_components=3, svd_solver="randomized")
    #rgb = pca.fit_transform(embeds)         # (N, 3)
    #rgb -= rgb.min(axis=0, keepdims=True)
    #rgb /= rgb.max(axis=0, keepdims=True) + 1e-7
    #rgb = (rgb * 255).astype(np.uint8)

    w0, h0  = slide.level_dimensions[0]
    nx = w0 // PATCH_SIZE
    ny = h0 // PATCH_SIZE
    draw_size = PATCH_SIZE // 16
    canvas = np.full((ny*draw_size, nx*draw_size, 3), 250, dtype=np.uint8)
    for (x, y), color in zip(coords, rgb):
        canvas[y:y+draw_size, x:x+draw_size] = color

    img = Image.fromarray(canvas)
    img.save(OUT_PATH)
    print("saved:", OUT_PATH)

def main():
    #embeds = get_embeds()
    data = np.load(f"{CASE}_embeds.npz")
    embeds = data["embeds"]
    pca(embeds)

if __name__ == "__main__":
    main()

