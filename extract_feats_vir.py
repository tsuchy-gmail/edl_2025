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
start = time.time()

vir = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
device = torch.device("cuda:0")
vir = vir.eval().to(device)
for p in vir.parameters(): 
    p.requires_grad = False

@torch.no_grad()                                      # ← 明示推論
def embed(x):                                         # x: (B,3,224,224)
    y = vir(x)                                        # (B,257,1280)
    feats = torch.cat([y[:,0], y[:,1:].mean(1)], -1)  # (B,2560)
    return feats                                      # on GPU

# ---- 2. Dataset ----
class Crop224DS(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        cfg = vir.pretrained_cfg
        self.tf = Compose([
            #Resize((224, 224)),
            CenterCrop(224),
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

def extract_feats(csvname, savename):
    img_paths = get_list_data(f"csv/{csvname}.csv")
    ds  = Crop224DS(img_paths)
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

    np.save(f"saved_feats/{savename}.npy", feat_mat)
    print("saved:", feat_mat.shape)
    end = time.time()
    print("time", f"{end-start:.4f}")

extract_feats("train_data_inside", "512center224_in_train")
