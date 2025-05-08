import cv2
import openslide
import numpy as np
import torch
import os
import random
import shutil

svs_root_dir = "/Raw/Kurume_Dataset/JMR_svs"
mask_root_dir = "/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/mask"
subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a", "FL/G3b"]

#case_names = [case_name.split("_")[0] for case_name in os.listdir(mask_root_dir) if not case_name.startswith(".")]

outside_ratio = 1
n_crop = 1000

crop_size = 512
stride = 512
layer = 2
resolutions = [1, 4, 16]

down_ratio = resolutions[layer]
crop_size_d = crop_size // down_ratio #(画像の解像度に注意)
stride_d = stride // down_ratio #(画像の解像度に注意)
saturation = 10
#th_saturation = saturation / 100 
tissue_ratio = 1


"""

'''
WSI(svsの読み込み)
svsは階層的なフォーマット
・0階層: x1
・1階層: x4
・2階層: x16

level_demensions: 各階層での画像サイズを取得する関数
read_region:　所望の領域を切り出す関数
- ((left top), layer, (crop_size, crop_size))
'''

for case_name in case_names:
    filename = svs_root_dir + "/" + case_name + ".svs"
    svs_image = openslide.OpenSlide(filename)
    width, height = svs_image.level_dimensions[0]
    width_2, height_2 = svs_image.level_dimensions[layer]
    print(width, height)
    print(width_2, height_2)

    excessX_2, excessY_2 = width_2 % stride_d, height_2 % stride_d #右と下のあまりを計算
    img = np.array(svs_image.read_region((0, 0), layer, (width_2 - excessX_2, height_2 - excessY_2)).convert("RGB"))
              
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#RGB to BGR
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)#HSV画像に変換
    print(np.max(hsv_img[:, :, 1]), np.min(hsv_img[:, :, 1]))
    del img #(メモリ制約のため入れることを推奨)

    '''
    - HSV空間で閾値処理 (1なら組織，0なら背景)
    ->平均プーリングで圧縮（0～1の値になる．視野内がすべて組織なら1,半々なら0.5，全て背景なら0）
    ->組織の割合で閾値処理

    binary_mapがマスク
    '''

    binary_map = (1.*(hsv_img[:,:, 1] >= saturation)).astype(np.float32)#閾値処理（いろんな処理があっていい）
#これ1.は必要？
    del hsv_img #(メモリ制約のため入れることを推奨)

    downsampled_map = torch.nn.functional.avg_pool2d(torch.from_numpy(binary_map).unsqueeze(0).unsqueeze(0), kernel_size=crop_size_d, stride=stride_d, padding=0)
    downsampled_map = (downsampled_map.squeeze() >= tissue_ratio)
    coors = torch.where(downsampled_map) #Trueのindexを取得
#del downsampled_map #(メモリ制約のため入れることを推奨)
    x_list = (coors[1]*stride).tolist() #ダウンサンプルされた座標にstrideをかけて元の画像の座標値とする
    y_list = (coors[0]*stride).tolist() #ダウンサンプルされた座標にstrideをかけて元の画像の座標値とする
    save_dir = f"/hyades/tsuchimoto/sat_test_imgs/sat_{saturation}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{case_name}.png"
    cv2.imwrite(save_path, 255 * binary_map.astype(np.uint8))

"""
def get_tissue_region_map_by_case(case_name):
    svs_path = svs_root_dir + "/" + case_name + ".svs"
    svs_img = openslide.OpenSlide(svs_path)
    width, height = svs_img.level_dimensions[layer]
    remainder_x, remainder_y = width % stride_d, height % stride_d
    whole_img = svs_img.read_region((0, 0), layer, (width - remainder_x, height - remainder_y)).convert("RGB")
    np_img = np.array(whole_img)
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2HSV_FULL)
    print(np.max(hsv_img[:, :, 1]), np.min(hsv_img[:, :, 1])) #これ何？

    del whole_img
    del np_img
    del np_img_bgr

    binary_map = (1.*(hsv_img[:,:,1] >= saturation))

    return binary_map

def get_follicle_region_map_by_case(subtype, case_name):
    tif_path = f"{mask_root_dir}/{subtype}/{case_name}.tif"
    mask_img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(mask_img, 20, 255, cv2.THRESH_BINARY)
    downsampled_map = torch.nn.functional.avg_pool2d(torch.from_numpy(bin_img).unsqueeze(0).unsqueeze(0).float(), kernel_size=4, stride=4, padding=0)

    downsampled_map = (downsampled_map.squeeze() >= 1).numpy()

    height, width = downsampled_map.shape
    adjusted_h = (height // stride_d) * stride_d
    adjusted_w = (width // stride_d) * stride_d

    '''
    save_dir = f"/hyades/tsuchimoto/binary_map"
    save_path = f"{save_dir}/down.png"
    #cv2.imwrite(save_path, 255 * downsampled_map.astype(np.uint8))
    print(adjusted_h, adjusted_w)
    '''
    

    return downsampled_map[:adjusted_h, :adjusted_w]

def get_outside_follicles_map_by_case(subtype, case_name):
    tissue_map = get_tissue_region_map_by_case(case_name)
    follicle_map = get_follicle_region_map_by_case(subtype, case_name)
    outside_map = tissue_map - follicle_map

    return outside_map


def write_binary_map(binary_map, case_name):
    save_dir = f"/hyades/tsuchimoto/outside_follicles"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{case_name}"
    cv2.imwrite(save_path, 255 * binary_map.astype(np.uint8))


def get_coordinates_by_case(subtype, case_name):
    outside_bin_map = get_outside_follicles_map_by_case(subtype, case_name)
    down_map = torch.nn.functional.avg_pool2d(torch.from_numpy(outside_bin_map).unsqueeze(0).unsqueeze(0), kernel_size=crop_size_d, stride=stride_d, padding=0)
    down_map_bool = (down_map.squeeze() >= outside_ratio)
    coordinates = torch.where(down_map_bool)

    x_list = (coordinates[1] * stride).tolist()
    y_list = (coordinates[0] * stride).tolist()

    return zip(x_list, y_list)

def crop_outside_random(subtype, case_name):
    crop_layer = 0
    xy_zipped = get_coordinates_by_case(subtype, case_name)
    xy_list = list(xy_zipped)
    random_xy_list = random.sample(xy_list, min(n_crop, len(xy_list)))

    svs_path = f"{svs_root_dir}/{case_name}.svs"
    svs_img = openslide.OpenSlide(svs_path)

    save_root_dir = f"/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size512_stride256/{subtype}"
    save_dir = f"{save_root_dir}/{case_name}/outside_follicles_n{n_crop}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for x, y in random_xy_list:
        cropped_img = svs_img.read_region((x, y), crop_layer, (crop_size, crop_size)).convert("RGB")
        filename = "img_x" + str(x) + "_y" + str(y) + ".tif"
        save_path = f"{save_dir}/{filename}"
        cropped_img.save(save_path)

def get_case_list(tif_path):
    return [tif.split(".")[0] for tif in os.listdir(tif_path) if tif.startswith("JMR")]
    

def main():
    for subtype in subtypes:
        case_list = get_case_list(f"{mask_root_dir}/{subtype}")
        for case in case_list:
            crop_outside_random(subtype, case)

main()
