import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import openslide
import torch
from tifffile import imread


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def dump_json(data, name):
    with open(name, "w") as f:
        json.dump(data, f, indent=4)

def vector_to_rgb(a, b):
    """
    a: 明るさの制御 (0 ~ 1, 小さいほど明るい)
    b: 色相の制御 (0 ~ 1, 小さいほど青、値が大きくなるほど赤)
    """
    # a を明るさ (Value) に反映: 1 - a (小さいほど明るい)
    value = 1 - a

    # b を色相 (Hue) に反映: 0 (青) ~ 240 (赤寄り)
    hue = b * 240  # OpenCVのHSV空間で0は青、240は赤寄り

    # 彩度は常に最大
    saturation = np.ones_like(hue)

    # HSV 配列を生成 (H, W, 3)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.float32)

    # OpenCVの形式に変換 (H: 0~360 → 0~180 にスケーリング)
    hsv[..., 0] = hsv[..., 0] / 2  # OpenCVはHの範囲が0~180
    hsv[..., 1:] *= 255  # S, V の範囲を0~1 → 0~255にスケーリング

    # HSV → RGB に変換
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    size = (1568, 3584) 
    resized = cv2.resize(bgr, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/hyades/tsuchimoto/bgr_as_cv2_resized_interp.png", resized)

    return rgb

def save_plt():
    cmap_unc = "viridis"
    cmap_p = "seismic"
    ones = np.ones_like(unc_map[..., 1])

    plt.figure(figsize=(10,18))
    plt.title("Confidence")
    plt.imshow(ones - unc_map[..., 0], cmap=cmap_unc)
    plt.colorbar()
    plt.savefig(f"/hyades/tsuchimoto/{case}_unc_hm_plt_{cmap_unc}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.title("PP of FL")
    plt.imshow(unc_map[..., 1], cmap=cmap_p)
    plt.colorbar()
    plt.savefig(f"/hyades/tsuchimoto/{case}_p_fl_hm_plt_{cmap_p}.png", dpi=300, bbox_inches="tight")


def overlay(img_np, wsi_path, a=5):
    ones = np.ones_like(img_np)
    img = (ones - img_np) * 255
    img = img.astype(np.uint8)
    spe_size = size
    print(img.shape)
    resized = cv2.resize(img, spe_size)
    color_img = cv2.applyColorMap(resized, cv2.COLORMAP_HOT)
    print("heatmap.shape",color_img.shape)
    
    #mask_path = "/Dataset/Kurume_Dataset/Annotated_Follicle_Dataset/2023/mask_x4/JMR1022_Reactive.tif"

    wsi = cv2.imread(wsi_path)
    #wsi = cv2.imread("/hyades/tsuchimoto/whole_img_lev2_1022.png")
    print("wsi.shape", wsi.shape)
    alpha = a / 10
    overlaid = cv2.addWeighted(wsi, alpha, color_img, 1-alpha, 0)
    cv2.imwrite(f"/hyades/tsuchimoto/{case}_overlay_alpha{a}.png", overlaid)



def overlay_follicles(case):
    mask_dir = "/Dataset/Kurume_Dataset/Follicular_Mask/2024_copy_from_tanaka/mask_x4/"
    mask_filelist = os.listdir(mask_dir)
    target_file = [mask_dir + file for file in mask_filelist if case in file][0]
    tif = imread(target_file)

    ones = np.ones_like(unc_map[..., 0])
    img = (ones - unc_map[..., 0]) * 255
    img = img.astype(np.uint8)
    spe_size = (img.shape[1] * 4, img.shape[0] * 4)

    color_img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    color_img = cv2.resize(color_img, spe_size)
    print("color_img.shape", color_img.shape)
    print("tif.shape", tif.shape)

    resized = cv2.resize(tif, spe_size)
    print("resised", resized.shape)
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours_len", len(contours))

    line_color = "white"
    line_colors = {"cyan": (255, 255, 0), "magenta": (255, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    thickness = 2
    color_img2 = cv2.drawContours(color_img, contours, -1, line_colors[line_color], thickness, cv2.LINE_8)
    cv2.imwrite(f"hm_{case}_color-{line_color}_wsicolor-viridis_thickness-{thickness}.png", color_img2)

case_idx = 0
cases = ["1022", "0072"]
case = cases[case_idx]


size_idx = case_idx
sizes = [(1750, 3778), (2345, 4457)]
size = sizes[size_idx]
path = f"unc_map_{case}.json"
unc_map = load_json(path)
unc_map = np.array(unc_map)


#overlay_follicles(case)
#save_plt()

#save_cv2(unc_map[..., 0])

#img_rgb = vector_to_rgb(unc_map[..., 0], unc_map[..., 1])
#wsi_path = f"/hyades/tsuchimoto/wsi{case}_lev2_org.png"
#overlay(unc_map[..., 0], wsi_path, a=9)

