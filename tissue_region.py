from openslide import OpenSlide
import numpy as np
import cv2
import PIL


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
'''
vis_img = wsi_np.copy()
for (x, y) in coords:
    cv2.circle(vis_img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

PIL.Image.fromarray(vis_img).save(f"figure/{case}_with_coords_{int(ratio*100)}per.png")
'''

case = "JMR0077"
wsi_path = f"/Raw/Kurume_Dataset/JMR_svs/{case}.svs"
slide = OpenSlide(wsi_path)
#coords_16x = tissue_region_coords_16x(slide)

def save_lev2(slide):
    level = 2
    org_size = 224
    div_scales = [1, 4, 16]
    div_scale = div_scales[level]
    ratio = 0.5

    start_tup = (0, 0)
    end_tup = slide.level_dimensions[level]

    wsi_lev2 = slide.read_region(start_tup, level, end_tup)
    wsi_lev2 = wsi_lev2.convert("RGB")
    wsi_lev2.save(f"figure/{case}_lev2.png")
save_lev2(slide)
