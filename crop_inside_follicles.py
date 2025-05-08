import cv2
import numpy as np
import matplotlib.pyplot as plt 

import torch

import os
import openslide

follicle_ratio = 1.0
stride = 256
crop_size = 512
down_ratio = 4
stride_d = stride // down_ratio
crop_size_d = crop_size // down_ratio #mask画像は4分の1

svs_root_dir = "/Raw/Kurume_Dataset/JMR_svs"
mask_root_dir = "/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/mask"

subtypes = ["Reactive", "FL/G1", "FL/G2", "FL/G3a", "FL/G3b"]

for target_subtype in subtypes:
    tif_list = os.listdir(f"{mask_root_dir}/{target_subtype}")
    save_root_dir = f"/Dataset/Kurume_Dataset/tsuchimoto/data/Follicle_Dataset/size{crop_size}_stride{stride}/{target_subtype}"
    cases = [tif.split(".")[0] for tif in tif_list if tif.startswith("JMR")]

    for target_case in cases:
        save_path = f"{save_root_dir}/{target_case}"

        if os.path.exists(save_path)
            continue

        mask_path = f"{mask_root_dir}/{target_subtype}/{target_case}.tif"
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        #2値化
        _, bin_img = cv2.threshold(mask_img, 20, 255, cv2.THRESH_BINARY)
        n_labels, id_map = cv2.connectedComponents(bin_img)
        print("n_labels", n_labels)

        
        os.makedirs(save_path, exist_ok = True)
        np.save(f"{save_path}/id_map.npy", id_map)
        plt.imshow(id_map) #pltでid_mapを見やすく可視化
        plt.savefig(f"{save_path}/id_map.png")
        plt.close()

        sum = 0

        for id in range(n_labels):
            if id == 0:
                continue

            binary_map = (id_map == id).astype(np.float32)
            downsampled_map = torch.nn.functional.avg_pool2d(torch.from_numpy(binary_map).unsqueeze(0).unsqueeze(0), kernel_size=crop_size_d, stride=stride_d, padding=0)
            #unsqueeze2回はavg_pool()が4階テンソルじゃないと使えないから
            downsampled_map = (downsampled_map.squeeze() >= follicle_ratio) #4階テンソルを元に戻して要素ごとにfollicle_ratioと比較
            coors = torch.where(downsampled_map) #Trueのindexを取得

            num = coors[0].shape[0]
            print("num",num)
            sum += num

            x_list = (coors[1] * stride).tolist()
            y_list = (coors[0] * stride).tolist()

            svs_path = f"{svs_root_dir}/{target_case}.svs"
            svs = openslide.OpenSlide(svs_path)
            os.makedirs(os.path.join(save_path, "follicle" + str(id)), exist_ok=True)

            for x, y in zip(x_list, y_list):
                crop_img = svs.read_region((x, y), 0, (crop_size, crop_size)).convert("RGB") #0は等倍という意味0階層目
                crop_img.save(os.path.join(save_path, "follicle" + str(id), "img_x" + str(x) + "_y" + str(y) + ".tif"))

        print("sum:", sum)




"""
for mask_path, svs_path, case in zip(mask_path_list, svs_path_list, case_list):

    if os.path.exists(save_root_dir + case):
        continue

    print("mask_path", mask_path)
    print("svs_path", svs_path)
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print(image.shape)

    #2値化
    _, bin_img = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    n_labels, id_map = cv2.connectedComponents(bin_img)
    print("n_labels", n_labels)

    
    os.makedirs(save_root_dir + case, exist_ok = True)
    np.save(save_root_dir + case + "/id_map.npy", id_map)
    plt.imshow(id_map) #pltでid_mapを見やすく可視化
    plt.savefig(save_root_dir + case + "/id_map.png")
    plt.close()

    sum = 0

    for id in range(n_labels):
        if id == 0:
            continue

        binary_map = (id_map == id).astype(np.float32)
        downsampled_map = torch.nn.functional.avg_pool2d(torch.from_numpy(binary_map).unsqueeze(0).unsqueeze(0), kernel_size=crop_size_d, stride=stride_d, padding=0)
        #unsqueeze2回はavg_pool()が4階テンソルじゃないと使えないから
        downsampled_map = (downsampled_map.squeeze() >= follicle_ratio) #4階テンソルを元に戻して要素ごとにfollicle_ratioと比較
        coors = torch.where(downsampled_map) #Trueのindexを取得

        num = coors[0].shape[0]
        print("num",num)
        sum += num

        x_list = (coors[1] * stride).tolist()
        y_list = (coors[0] * stride).tolist()
        svs = openslide.OpenSlide(svs_path)
        os.makedirs(os.path.join(save_root_dir, case, "follicle" + str(id)), exist_ok=True)

        for x, y in zip(x_list, y_list):
            crop_img = svs.read_region((x, y), 0, (crop_size, crop_size)).convert("RGB") #0は等倍という意味0階層目
            crop_img.save(os.path.join(save_root_dir, case, "follicle" + str(id), "img_x" + str(x) + "_y" + str(y) + ".tif"))

    print("sum:", sum)
    exit(0)
    
    

64分の1の場合
まず、マスク画像の白い部分のみ1.0、他は0の2次元配列を作る - binary_map
avg_pool()で、64マスの和を取ってその平均を値とする1マスにダウン - downsampled_map  * strideがkernelより小さくても、strideが可能な分の出力が増える
downsampled_mapの値を1/0からTrue/Falseに変更
torch.whereでTrueの部分の座標を取得 - coors
"""
