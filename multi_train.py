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
from multiprocessing import Process

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

def to_avg_dict(epoch, loss, loss_in, loss_out, mse_in, mse_out, kl_in, kl_out, n_data, acc=None, acc_in=None, acc_out=None):
    n_data_half = n_data // 2
    if acc and acc_in and acc_out:
        return {
                "epoch": epoch + 1,
                "acc": acc,
                "acc_in": acc_in,
                "acc_out": acc_out,
                "loss": loss / n_data,
                "loss_in": loss_in / n_data_half,
                "loss_out": loss_out / n_data_half,
                "mse_in": mse_in / n_data_half,
                "mse_out": mse_out / n_data_half,
                "kl_in": kl_in / n_data_half,
                "kl_out": kl_out / n_data_half,
                }
    else:
        return {
                "epoch": epoch + 1,
                "loss": loss / n_data,
                "loss_in": loss_in / n_data_half,
                "loss_out": loss_out / n_data_half,
                "mse_in": mse_in / n_data_half,
                "mse_out": mse_out / n_data_half,
                "kl_in": kl_in / n_data_half,
                "kl_out": kl_out / n_data_half,
                }


def test(model, loader, records, save_ok, device, lamb, epoch, result_path, epochs):
    loss_total = 0.0
    loss_in_total = 0.0
    loss_out_total = 0.0
    mse_in_total = 0.0
    mse_out_total = 0.0
    kl_in_total = 0.0
    kl_out_total = 0.0

    n_correct = 0
    n_correct_in = 0
    n_correct_out = 0

    n_data_in = 0
    n_data_out = 0

    alpha_list_R_in = []
    alpha_list_R_out = []
    alpha_list_F_in = []
    alpha_list_F_out = []
    alpha_save_cond = (epoch+1) % 10 == 0 or epoch == 0
    with torch.no_grad():
        model.eval()
        for img_batch, label_batch, region_batch in loader:
            img_batch = img_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            region_batch = region_batch.to(device, non_blocking=True)
            alpha_batch = model(img_batch)


            loss = evidential_classification(alpha_batch, label_batch, lamb)
            loss_total += loss.item() * img_batch.size(0)
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            R_mask = label_batch == R
            F_mask = label_batch == F
            in_mask = region_batch == IN
            out_mask = region_batch == OUT

            R_in_mask = R_mask & in_mask
            R_out_mask = R_mask & out_mask
            F_in_mask = F_mask & in_mask
            F_out_mask = F_mask & out_mask

            if alpha_save_cond:
                alpha_list_R_in.append(alpha_batch[R_in_mask])
                alpha_list_R_out.append(alpha_batch[R_out_mask])
                alpha_list_F_in.append(alpha_batch[F_in_mask])
                alpha_list_F_out.append(alpha_batch[F_out_mask])

            n_data_in += in_mask.sum()
            n_data_out += out_mask.sum()
            
            n_correct += (alpha_batch.argmax(-1) == label_batch).sum()
            n_correct_in += (alpha_batch[in_mask].argmax(-1) == label_batch[in_mask]).sum()
            n_correct_out += (alpha_batch[out_mask].argmax(-1) == label_batch[out_mask]).sum()

            mse_in_total += mse_batch[in_mask].sum().item()
            mse_out_total += mse_batch[out_mask].sum().item()
            kl_in_total += kl_batch[in_mask].sum().item()
            kl_out_total += kl_batch[out_mask].sum().item()

            loss_in_total += (mse_batch[in_mask] + lamb * kl_batch[in_mask]).sum().item()
            loss_out_total += (mse_batch[out_mask] + lamb * kl_batch[out_mask]).sum().item()
    
    if alpha_save_cond:
        alpha_R_in = torch.cat(alpha_list_R_in, dim=0).cpu().numpy()
        alpha_R_out = torch.cat(alpha_list_R_out, dim=0).cpu().numpy()
        alpha_F_in = torch.cat(alpha_list_F_in, dim=0).cpu().numpy()
        alpha_F_out = torch.cat(alpha_list_F_out, dim=0).cpu().numpy()

        alpha_save_dir = os.path.join(result_path, "alpha", f"epoch{epoch+1}")
        os.makedirs(alpha_save_dir, exist_ok=True)
        pd.DataFrame(alpha_R_in).to_csv(os.path.join(alpha_save_dir, f"R_in.csv"), index=False)
        pd.DataFrame(alpha_R_out).to_csv(os.path.join(alpha_save_dir, f"R_out.csv"), index=False)
        pd.DataFrame(alpha_F_in).to_csv(os.path.join(alpha_save_dir, f"F_in.csv"), index=False)
        pd.DataFrame(alpha_F_out).to_csv(os.path.join(alpha_save_dir, f"F_out.csv"), index=False)

        npz_path = os.path.join(alpha_save_dir, "alpha.npz")
        np.savez(npz_path, R_in=alpha_R_in, R_out=alpha_R_out, F_in=alpha_F_in, F_out=alpha_F_out)
        alpha_list_R_in = []
        alpha_list_R_out = []
        alpha_list_F_in = []
        alpha_list_F_out = []

    n_data = len(loader.dataset)
    n_data_in = n_data_in.item()
    n_data_out = n_data_out.item()

    acc = (n_correct.item() / n_data)
    acc_in = (n_correct_in.item() / n_data_in)
    acc_out = (n_correct_out.item() / n_data_out)


    avg_dict = to_avg_dict(epoch, loss_total, loss_in_total, loss_out_total, mse_in_total, mse_out_total, kl_in_total, kl_out_total, n_data, acc, acc_in, acc_out)
    records.append(avg_dict)

    if save_ok:
        csv_save_path = os.path.join(result_path, "test_loss.csv")
        pd.DataFrame(records).to_csv(csv_save_path, index=False)

        torch.save(model.state_dict(), os.path.join(result_path, "model_last.pth"))

    return avg_dict["loss_in"], avg_dict["loss_out"], acc, acc_in, acc_out

def get_loader(transform, ini_num_workers, num_workers):
    train_img_path_list, train_subtype_list, train_region_list = get_list_data("csv/train_data.csv")
    train_img_list = preload_all_imgs(train_img_path_list, transform, ini_num_workers)
    train_ds = CustomDataset(train_img_list, train_subtype_list, train_region_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    test_img_path_list, test_subtype_list, test_region_list = get_list_data("csv/test_data.csv")
    test_img_list = preload_all_imgs(test_img_path_list, transform, ini_num_workers)
    test_ds = CustomDataset(test_img_list, test_subtype_list, test_region_list)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader

def train(train_loader, test_loader, epochs, cuda, save_ok, dir_suffix):
    start_time = time()
    print("training start")
    print(f"epochs: {epochs}")
    if save_ok:
        year = datetime.now().strftime("%Y")
        date = datetime.now().strftime("%m_%d")
        timestamp = datetime.now().strftime("%H%M_%S%f")[:-3]

        result_dir_path = f"result/{year}/{date}/{timestamp}_{dir_suffix}"
        os.makedirs(result_dir_path, exist_ok=True)

    device = torch.device(f"cuda:{cuda}")  

    print("preloading start")
    train_records = []

    model = models.resnet18(weights=None)
    n_features = model.fc.in_features
    model.fc = Dirichlet(n_features, n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_records = []
    min_test_loss_in = float("inf")
    min_test_loss_out = float("inf")
    max_acc = -1
    max_acc_in = -1
    max_acc_out = -1

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

        #lamb = 10 - ((epoch + 1) * 0.04)
        #lamb = (epoch + 1) / epochs
        lamb = 1
        for img_batch, label_batch, region_batch in train_loader:
            img_batch = img_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            region_batch = region_batch.to(device, non_blocking=True)
            alpha_batch = model(img_batch)

            loss = evidential_classification(alpha_batch, label_batch, lamb)
            loss_total += loss.item() * img_batch.size(0) #バッチ平均 * バッチサイズ = バッチの合計ロス
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            in_mask = region_batch == IN
            out_mask = region_batch == OUT

            mse_in_total += mse_batch[in_mask].sum().item()
            mse_out_total += mse_batch[out_mask].sum().item()
            kl_in_total += kl_batch[in_mask].sum().item()
            kl_out_total += kl_batch[out_mask].sum().item()

            loss_in_total += (mse_batch[in_mask] + lamb * kl_batch[in_mask]).sum().item()
            loss_out_total += (mse_batch[out_mask] + lamb * kl_batch[out_mask]).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        n_data = len(train_loader.dataset)

        if save_ok:
            avg_dict = to_avg_dict(epoch, loss_total, loss_in_total, loss_out_total, mse_in_total, mse_out_total, kl_in_total, kl_out_total, n_data)
            train_records.append(avg_dict)
            csv_save_path = os.path.join(result_dir_path, "train_loss.csv")
            pd.DataFrame(train_records).to_csv(csv_save_path, index=False)


        avg_test_loss_in, avg_test_loss_out, acc, acc_in, acc_out = test(model, test_loader, test_records, save_ok, device, lamb, epoch, result_dir_path, epochs)
        """

        if avg_test_loss_in < min_test_loss_in:
            min_test_loss_in = avg_test_loss_in
            torch.save(model.state_dict(), os.path.join(result_dir_path, "min_test_loss_in.pth"))
        if avg_test_loss_out < min_test_loss_out:
            min_test_loss_out = avg_test_loss_out
            torch.save(model.state_dict(), os.path.join(result_dir_path, "min_test_loss_out.pth"))

        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), os.path.join(result_dir_path, "max_acc.pth"))
        if acc_in > max_acc_in:
            max_acc_in = acc_in
            torch.save(model.state_dict(), os.path.join(result_dir_path, "max_acc_in.pth"))
        if acc_out > max_acc_out:
            max_acc_out = acc_out
            torch.save(model.state_dict(), os.path.join(result_dir_path, "max_acc_out.pth"))
        """


    end_time = time()
    elapsed_time = end_time - start_time
    elapsed_time = str(timedelta(seconds=elapsed_time))
    print("time", elapsed_time)
    time_path = os.path.join(result_dir_path, "time.txt")
    with open(time_path, "w") as f:
        f.write(elapsed_time)
        


#dir_suffix = f"epoch{epochs}"

def main():
    n_ps = int(sys.argv[1])
    cudas = [None] * n_ps
    for i in range(n_ps):
        cudas[i] = int(sys.argv[2+i])
    ini_num_workers = int(sys.argv[2 + n_ps]) if len(sys.argv) >= (3 + n_ps) else cpu_count-1
    num_workers = int(sys.argv[3+n_ps]) if len(sys.argv) >= (4+n_ps) else cpu_count-1
    n_loop = int(sys.argv[4+n_ps]) if len(sys.argv) >= (5+n_ps) else 1
    dir_suffix = sys.argv[5+n_ps] if len(sys.argv) >= (6+n_ps) else ""

    transform = get_transforms()
    train_loader, test_loader = get_loader(transform, ini_num_workers, num_workers)

    for _ in range(n_loop):
        ps_list = []
        for cuda in cudas:
            ps = Process(target=train, kwargs={
                "train_loader": train_loader,
                "test_loader": test_loader,
                "epochs": 100,
                "cuda": cuda,
                "save_ok": True,
                "dir_suffix": dir_suffix,
                })
            ps.start()  
            ps_list.append(ps)

        for ps in ps_list:
            ps.join()

main()

'''
train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=100,
        cuda=cuda, 
        transform=get_transforms(),
        save_dir="05_23",
        save_ok=True,
        ini_num_workers=ini_num_workers,
        num_workers=num_workers,
        dir_suffix=dir_suffix
     )
'''
