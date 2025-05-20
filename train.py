from torch.utils.data import Dataset, DataLoader
from PIL import Image 
from torchvision import transforms, models
import pandas as pd
import torch

from pandas import read_csv
from datetime import datetime
from edl_pytorch import Dirichlet, evidential_classification
import os
from my_loss_function import my_evidential_classification
import random
from tqdm import tqdm
import sys


REACTIVE = "Reactive"
FL = "FL"
INSIDE = "inside"
OUTSIDE = "outside"

IN = 1
OUT = 0

batch_size = 256
n_classes = 2
learning_rate = 1e-4
epochs = 100
cpu_count = os.cpu_count()

def encode_subtype(subtype):
    if subtype == REACTIVE:
        return torch.tensor(0, dtype=torch.long)
    elif FL in subtype:
        return torch.tensor(1, dtype=torch.long)
    else:
        raise ValueError("subtypeがReactiveでもFLでもない")

def encode_region(region):
    if region == OUTSIDE:
        return torch.tensor(0, dtype=torch.long)
    elif region == INSIDE:
        return torch.tensor(1, dtype=torch.long)
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

def preload_all_imgs(img_path_list, transform, ini_num_workers):
    img_ds = ImageDataset(img_path_list, transform)
    img_loader = DataLoader(img_ds, batch_size=batch_size, shuffle=False, num_workers=ini_num_workers)
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


class RandomRotation90:
    def __call__(self, img):
        angles = [90, 180, 270]
        angle = random.choice(angles)

        return transforms.functional.rotate(img, angle)

def get_transforms():
    crop_size = (224, 224)

    return transforms.Compose(
            [
                RandomRotation90(),
                transforms.CenterCrop(size=crop_size),
                transforms.ToTensor(),
            ]
    )

def to_avg_dict(epoch, loss, loss_in, loss_out, mse_in, mse_out, kl_in, kl_out, n_data):
    n_data_half = n_data // 2
    return {
            "epoch": epoch + 1,
            "loss": loss / n_data,
            "loss_in": loss_in / n_data_half,
            "loss_out": loss_out / n_data_half,
            "mse_in": mse_in / n_data_half,
            "mse_in": mse_out / n_data_half,
            "kl_in": kl_in / n_data_half,
            "kl_out": kl_out / n_data_half,
            }


def train(cuda, transform, save_dir, save_ok, ini_num_workers, num_workers):
    print("training start")
    print("ini_num_workers", ini_num_workers)
    print("num_workers", num_workers)
    if save_ok:
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        result_dir_path = f"result/{save_dir}/{timestamp}/"
        os.makedirs(result_dir_path, exist_ok=True)


    device = torch.device(f"cuda:{cuda}")  

    train_img_path_list, train_subtype_list, train_region_list = get_list_data("csv/train_data.csv")
    test_img_path_list, test_subtype_list, test_region_list = get_list_data("csv/test_data.csv")
    
    print("preloading start")

    train_img_list = preload_all_imgs(train_img_path_list, transform, ini_num_workers)
    test_img_list = preload_all_imgs(test_img_path_list, transform, ini_num_workers)

    train_ds = CustomDataset(train_img_list, train_subtype_list, train_region_list)
    test_ds = CustomDataset(test_img_list, test_subtype_list, test_region_list)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    model = models.resnet18(weights=None)
    n_features = model.fc.in_features

    model.fc = Dirichlet(n_features, n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_testloss_inside = float("inf")

    train_records = []
    for epoch in range(epochs):
        print(f"epoch{epoch + 1}")
        model.train()

        loss_total = 0.0
        loss_in_total = 0.0
        loss_out_total = 0.0
        mse_in_total = 0.0
        mse_out_total = 0.0
        kl_in_total = 0.0
        kl_out_total = 0.0

        lamb = 10 - ((epoch + 1) * 0.04)
        for img_batch, label_batch, region_batch in train_loader:
            img_batch = img_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            region_batch = region_batch.to(device, non_blocking=True)
            alpha_batch = model(img_batch)

            loss = evidential_classification(alpha_batch, label_batch, lamb)
            loss_total += loss.item() * img_batch.size(0) #バッチ平均 * バッチサイズ = バッチの合計ロス
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            inside_mask = region_batch == IN
            outside_mask = region_batch == OUT

            mse_in_total += mse_batch[inside_mask].sum().item()
            mse_out_total += mse_batch[outside_mask].sum().item()
            kl_in_total += kl_batch[inside_mask].sum().item()
            kl_out_total += kl_batch[outside_mask].sum().item()

            loss_in_total += (mse_batch[inside_mask] + lamb * kl_batch[inside_mask]).sum().item()
            loss_out_total += (mse_batch[outside_mask] + lamb * kl_batch[outside_mask]).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"n_in{n_in}, n_out{n_out}")
        n_data = len(train_loader.dataset)

        if save_ok:
            avg_dict = to_avg_dict(epoch, loss_total, loss_in_total, loss_out_total, mse_in_total, mse_out_total, kl_in_total, kl_out_total, n_data)
            train_records.append(avg_dict)
            csv_save_path = os.path.join(result_dir_path, "train_loss.csv")
            pd.DataFrame(train_records).to_csv(csv_save_path, index=False)
            print("write train_info to csv")

        #test
        loss_total = 0.0
        loss_in_total = 0.0
        loss_out_total = 0.0
        mse_in_total = 0.0
        mse_out_total = 0.0
        kl_in_total = 0.0
        kl_out_total = 0.0

        with torch.no_grad():
            model.eval()
            correct = 0
            correct_inside = 0
            correct_outside = 0
            total = 0

            for img_batch, label_batch, region_batch in train_loader:
                img_batch = img_batch.to(device, non_blocking=True)
                label_batch = label_batch.to(device, non_blocking=True)
                region_batch = region_batch.to(device, non_blocking=True)
                alpha_batch = model(img_batch)

                loss = evidential_classification(alpha_batch, label_batch, lamb)
                loss_total += loss.item() * img_batch.size(0)
                mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

                inside_mask = region_batch == IN
                outside_mask = region_batch == OUT

                mse_in_total += mse_batch[inside_mask].sum().item()
                mse_out_total += mse_batch[outside_mask].sum().item()
                kl_in_total += kl_batch[inside_mask].sum().item()
                kl_out_total += kl_batch[outside_mask].sum().item()

                loss_in_total += (mse_batch[inside_mask] + lamb * kl_batch[inside_mask]).sum().item()
                loss_out_total += (mse_batch[outside_mask] + lamb * kl_batch[outside_mask]).sum().item()

        n_data = len(test_loader.dataset)
        avg_loss = loss_total / n_data
        avg_loss_inside = loss_inside / (n_data // 2)
        avg_loss_outside = loss_outside / (n_data // 2)

        avg_mse_inside = mse_inside_total / (n_data // 2)
        avg_mse_outside = mse_outside_total / (n_data // 2)
        
        avg_kl_inside = kl_inside_total / (n_data // 2)
        avg_kl_outside = kl_outside_total / (n_data // 2)

        loss_list_test.extend([avg_loss])
        loss_inside_list_test.extend([avg_loss_inside])
        loss_outside_list_test.extend([avg_loss_outside])
        mse_inside_list_test.extend([avg_mse_inside])
        mse_outside_list_test.extend([avg_mse_outside])
        kl_inside_list_test.extend([avg_kl_inside])
        kl_outside_list_test.extend([avg_kl_outside])
        print("avg_loss_test", avg_loss)

        loss_data_for_csv = {
                "epoch": epoch_list,
                "loss": loss_list_test,
                "loss_inside": loss_inside_list_test,
                "loss_outside": loss_outside_list_test,
                "mse_inside": mse_inside_list_test,
                "mse_outside": mse_outside_list_test,
                "kl_inside": kl_inside_list_test,
                "kl_outside": kl_outside_list_test,
                }

        loss_df = pd.DataFrame(loss_data_for_csv)
        if save_ok:
            csv_path = os.path.join(result_dir_path, "test_loss_status.csv")
            loss_df.to_csv(csv_path, index=False)
            print("write test_info to csv")

            torch.save(model.state_dict(), os.path.join(result_dir_path, "model_last_epoch.pth"))

            if avg_loss_inside < min_testloss_inside:
                min_testloss_insdie = avg_loss_inside
                torch.save(model.state_dict(), os.path.join(result_dir_path, "min_testloss_inside.pth"))


print(len(sys.argv))

ini_num_workers = cpu_count - 1
num_workers = cpu_count - 1
cuda = int(sys.argv[1])

if len(sys.argv) >= 3:
    ini_num_workers = int(sys.argv[2])
if len(sys.argv) >= 4:
    num_workers = int(sys.argv[3])

train(
        cuda=cuda, 
        transform=get_transforms(),
        save_dir="05_20",
        save_ok=True,
        ini_num_workers=ini_num_workers,
        num_workers=num_workers,
     )
