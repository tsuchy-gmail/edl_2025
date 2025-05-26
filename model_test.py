from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models 
import pandas as pd
import torch
from pandas import read_csv
from datetime import datetime, timedelta
from time import time
from edl_pytorch import Dirichlet, evidential_classification
import os
from my_loss_function import my_evidential_classification
import random
from tqdm import tqdm
import sys
import numpy as np

REACTIVE = "Reactive"
FL = "FL"
R = 0
F = 1
INSIDE = "inside"
OUTSIDE = "outside"
OUT = 0
IN = 1
batch_size = 256
n_classes = 2
learning_rate = 1e-4
cpu_count = os.cpu_count()

def test(model, loader, device, result_path, epochs):
    alpha_list_R_in = []
    alpha_list_R_out = []
    alpha_list_F_in = []
    alpha_list_F_out = []

    model.to(device)
    with torch.no_grad():
        model.eval()
        for img_batch, label_batch, region_batch in loader:
            img_batch = img_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            region_batch = region_batch.to(device, non_blocking=True)
            alpha_batch = model(img_batch)

            R_mask = label_batch == R
            F_mask = label_batch == F
            in_mask = region_batch == IN
            out_mask = region_batch == OUT

            R_in_mask = R_mask & in_mask
            R_out_mask = R_mask & out_mask
            F_in_mask = F_mask & in_mask
            F_out_mask = F_mask & out_mask

            alpha_list_R_in.append(alpha_batch[R_in_mask])
            alpha_list_R_out.append(alpha_batch[R_out_mask])
            alpha_list_F_in.append(alpha_batch[F_in_mask])
            alpha_list_F_out.append(alpha_batch[F_out_mask])

    alpha_R_in = torch.cat(alpha_list_R_in, dim=0).cpu().numpy()
    alpha_R_out = torch.cat(alpha_list_R_out, dim=0).cpu().numpy()
    alpha_F_in = torch.cat(alpha_list_F_in, dim=0).cpu().numpy()
    alpha_F_out = torch.cat(alpha_list_F_out, dim=0).cpu().numpy()

    alpha_save_dir = os.path.join(result_path, "alpha_last")
    os.makedirs(alpha_save_dir, exist_ok=True)
    pd.DataFrame(alpha_R_in).to_csv(os.path.join(alpha_save_dir, f"R_in.csv"), index=False)
    pd.DataFrame(alpha_R_out).to_csv(os.path.join(alpha_save_dir, f"R_out.csv"), index=False)
    pd.DataFrame(alpha_F_in).to_csv(os.path.join(alpha_save_dir, f"F_in.csv"), index=False)
    pd.DataFrame(alpha_F_out).to_csv(os.path.join(alpha_save_dir, f"F_out.csv"), index=False)

    npz_path = os.path.join(alpha_save_dir, "alpha.npz")
    np.savez(npz_path, R_in=alpha_R_in, R_out=alpha_R_out, F_in=alpha_F_in, F_out=alpha_F_out)
    print("saved to", alpha_save_dir)


def encode_subtype(subtype):
    if subtype == REACTIVE:
        return torch.tensor(R, dtype=torch.long)
    elif FL in subtype:
        return torch.tensor(F, dtype=torch.long)
    else:
        raise ValueError("subtypeがReactiveでもFLでもない")

def encode_region(region):
    if region == OUTSIDE:
        return torch.tensor(OUT, dtype=torch.long)
    elif region == INSIDE:
        return torch.tensor(IN, dtype=torch.long)
    else:
        raise ValueError("regionに想定外の値")

class ImageDataset(Dataset):
    def __init__(self, img_path_list, transform=None):
        self.img_path_list = img_path_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img

def preload_all_imgs(img_path_list, transform):
    img_ds = ImageDataset(img_path_list, transform)
    img_loader = DataLoader(img_ds, batch_size=batch_size, shuffle=False, num_workers=cpu_count-1)
    tensor_img_list = []
    for i, img_batch in enumerate(tqdm(img_loader, desc="Loading imgs in batch")):
        print(f"{i+1} / {len(img_loader)}")
        tensor_img_list.append(img_batch)
    all_imgs = torch.cat(tensor_img_list, dim=0)
    print("all_imgs.shape", all_imgs.shape)

    return all_imgs

class CustomDataset(Dataset):
    def __init__(self, tensor_img_list, subtype_list, region_list):
        self.tensor_img_list = tensor_img_list
        self.subtype_list = subtype_list
        self.region_list = region_list

    def __len__(self):
        return self.tensor_img_list.size(0)
    
    def __getitem__(self, idx):
        img = self.tensor_img_list[idx]
        subtype = self.subtype_list[idx]
        label = encode_subtype(subtype)
        region = self.region_list[idx]
        region = encode_region(region)

        return img, label, region

def get_list_data(csv_path):
    data_df = read_csv(csv_path)

    img_path_list = data_df["img_path"].tolist()
    subtype_list = data_df["subtype"].tolist()
    region_list = data_df["region"].tolist()

    return img_path_list, subtype_list, region_list
def get_transforms():
    crop_size = (224, 224)

    return transforms.Compose(
            [
                transforms.CenterCrop(size=crop_size),
                transforms.ToTensor(),
            ]
    )

def get_loader():
    transform = get_transforms()
    test_img_path_list, test_subtype_list, test_region_list = get_list_data("csv/test_data.csv")
    test_img_list = preload_all_imgs(test_img_path_list, transform)
    test_ds = CustomDataset(test_img_list, test_subtype_list, test_region_list)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=cpu_count-1)

    return test_loader

def load_model(model_path, cuda):
    model = models.resnet18(weights=None)
    n_features = model.fc.in_features
    model.fc = Dirichlet(n_features, n_classes)
    model.load_state_dict(torch.load(model_path, map_location=f"cuda:{cuda}"))

    return model
def main():
    epochs = 100
    cuda = int(sys.argv[1])
    device = torch.device(f"cuda:{cuda}")
    loader = get_loader()
    target_dates = ["05_21", "05_23"]
    for target_date in target_dates:
        model_root_dir = os.path.join("result", target_date)
        model_dir_list = os.listdir(model_root_dir)
        for model_dir in model_dir_list:
            target_model_dir = os.path.join(model_root_dir, model_dir)
            alpha_dir = os.path.join(target_model_dir, "alpha_last")
            if os.path.exists(alpha_dir):
                continue
            csv_path = os.path.join(target_model_dir, "train_loss.csv")
            if not os.path.exists(csv_path):
                continue
            csv_df = pd.read_csv(csv_path)
            if len(csv_df) == epochs: #学習が正しく終了していれば
                model_path = os.path.join(target_model_dir, "model_last.pth")
                model = load_model(model_path, cuda)
                test(model, loader, device, target_model_dir, epochs)


        
if __name__ == "__main__":
    main()

        

