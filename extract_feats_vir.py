import torch, timm, numpy as np, pathlib
from timm.layers import SwiGLUPacked
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision.transforms import CenterCrop, Normalize, Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image 
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
start = time.time()

vir = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
device = torch.device("cuda:1")
vir = vir.eval().to(device)
for p in vir.parameters(): 
    p.requires_grad = False

@torch.no_grad()                                      # ← 明示推論
def embed(x):                                         # x: (B,3,224,224)
    y = vir(x)                                        # (B,257,1280)
    feats = torch.cat([y[:,0], y[:,1:].mean(1)], -1)  # (B,2560)
    #feats = torch.cat([y[:,0]], -1) #only CLS token
    #feats = y[:,1:].mean(1) #only 14patch token
    return feats                                      # on GPU

# ---- 2. Dataset ----
class Crop224DS(Dataset):
    def __init__(self, img_paths, org_size):
        self.img_paths = img_paths
        cfg = vir.pretrained_cfg
        self.tf = Compose([
            CenterCrop(org_size),
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=cfg["mean"], std=cfg["std"])
        ])
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.tf(img)

# 例: png とラベル csv があると仮定
def get_list_data(csv_path):
    data_df = pd.read_csv(csv_path)
    img_path_list = data_df["img_path"].tolist()

    return img_path_list

def show_img9(img_batch):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(F.to_pil_image(img_batch[i]))  # PILに変換して表示
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("result/resized9.png")
    exit(0)

def extract_feats(org_size, tar_dir, filename, savename):
    img_paths = get_list_data(f"csv/{tar_dir}{filename}")
    ds  = Crop224DS(img_paths, org_size)
    ldr = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True, shuffle=False)

    N = len(ds)
    feat_mat = np.empty((N, 2560), dtype="float32")
    idx = 0


    for img_batch in tqdm(ldr, total=len(ldr)):
        img_batch = img_batch.to(device, non_blocking=True)
        #show_img9(img_batch)
        feats = embed(img_batch).cpu().numpy()
        print(feats.shape)
        feat_mat[idx:idx+len(feats)] = feats
        idx += len(feats)

    save_root_dir = "saved_feats/"
    save_dir = os.path.join(save_root_dir, tar_dir)
    save_path = os.path.join(save_dir, savename)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_path, feat_mat)
    print("saved:", feat_mat.shape)
    end = time.time()
    print("time", f"{end-start:.4f}")

patch_size = 896
stride = 896
#tar_dir = f"size{patch_size}_stride{stride}"
tar_dir = ""
org_size = 448
extract_feats(org_size, tar_dir, "train_data_inside.csv", "448Res224_in_train.npy")
#extract_feats(org_size, tar_dir, "test_data.csv", "448Res224_test.npy")
