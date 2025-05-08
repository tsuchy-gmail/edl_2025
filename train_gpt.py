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

# 定数定義
REACTIVE = "Reactive"
FL = "FL"
INSIDE = "inside"
OUTSIDE = "outside"
batch_size = 128
n_classes = 2
learning_rate = 1e-4
epochs = 100
num_workers = 8

# サブタイプを数値ラベルに変換
def encode_subtype(subtype):
    if subtype == REACTIVE:
        return 0
    elif FL in subtype:
        return 1
    else:
        raise ValueError("subtype が Reactive でも FL でもない")

# リージョンを数値ラベルに変換 (inside=1, outside=0)
def encode_region(region):
    if region == INSIDE:
        return 1
    elif region == OUTSIDE:
        return 0
    else:
        raise ValueError("region に想定外の値")

# データセット定義
class CustomDataset(Dataset):
    def __init__(self, img_path_list, subtype_list, region_list, transform=None):
        self.img_path_list = img_path_list
        self.subtype_list = subtype_list
        self.region_list = region_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        # 画像読み込み
        img = Image.open(self.img_path_list[idx]).convert("RGB")
        # ラベル・リージョンを数値化しテンソル化
        label_val = encode_subtype(self.subtype_list[idx])
        region_val = encode_region(self.region_list[idx])
        label = torch.tensor(label_val, dtype=torch.long)
        region = torch.tensor(region_val, dtype=torch.bool)

        if self.transform:
            img = self.transform(img)
        return img, label, region

# CSV からリストを取得
def get_list_data(csv_path):
    data_df = read_csv(csv_path)
    img_path_list = data_df["img_path"].tolist()
    subtype_list = data_df["subtype"].tolist()
    region_list = data_df["region"].tolist()
    return img_path_list, subtype_list, region_list

# ランダム回転
class RandomRotation90:
    def __call__(self, img):
        angles = [90, 180, 270]
        angle = random.choice(angles)
        return transforms.functional.rotate(img, angle)

# 前処理定義
def get_transforms():
    crop_size = (224, 224)
    return transforms.Compose([
        RandomRotation90(),
        transforms.CenterCrop(size=crop_size),
        transforms.ToTensor(),
    ])

# 訓練・評価ループ
def train(cuda, transforms, save_dir):
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
    result_dir = f"result/{save_dir}/{timestamp}/"
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device(f"cuda:{cuda}")
    train_imgs, train_subs, train_regs = get_list_data("csv/train_data.csv")
    test_imgs,  test_subs,  test_regs  = get_list_data("csv/test_data.csv")

    train_ds = CustomDataset(train_imgs, train_subs, train_regs, transforms)
    test_ds  = CustomDataset(test_imgs,  test_subs,  test_regs,  transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=num_workers)

    model = models.resnet18(pretrained=False)
    n_features = model.fc.in_features
    model.fc = Dirichlet(n_features, n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ログ用辞書
    train_logs = {k: [] for k in ["loss", "loss_in", "loss_out", "mse_in", "mse_out", "kl_in", "kl_out"]}
    test_logs  = {k: [] for k in ["loss", "loss_in", "loss_out", "mse_in", "mse_out", "kl_in", "kl_out"]}

    for epoch in range(epochs):
        lamb = 5 - ((epoch + 1) * 0.04)

        # 訓練フェーズ
        model.train()
        sum_loss = 0.0
        acc = {k: torch.tensor(0.0, device=device) for k in train_logs if k != "loss"}

        for imgs, labels, regs in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            regs   = regs.to(device, non_blocking=True)

            alpha = model(imgs)
            loss  = evidential_classification(alpha, labels, lamb)
            sum_loss += loss.item() * imgs.size(0)

            mse, kl = my_evidential_classification(alpha, labels)
            acc["mse_in"]  += mse[regs].sum()
            acc["kl_in"]   += kl[regs].sum()
            acc["mse_out"] += mse[~regs].sum()
            acc["kl_out"]  += kl[~regs].sum()
            acc["loss_in"]  += mse[regs].sum()      + lamb * kl[regs].sum()
            acc["loss_out"] += mse[~regs].sum()    + lamb * kl[~regs].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 訓練ログ計算
        n_train = len(train_loader.dataset)
        train_logs["loss"].append(sum_loss / n_train)
        for key in acc:
            train_logs[key].append(acc[key].item() / n_train)

        # テストフェーズ
        model.eval()
        with torch.no_grad():
            sum_loss = 0.0
            acc = {k: torch.tensor(0.0, device=device) for k in test_logs if k != "loss"}

            for imgs, labels, regs in test_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                regs   = regs.to(device, non_blocking=True)

                alpha = model(imgs)
                loss  = evidential_classification(alpha, labels, lamb)
                sum_loss += loss.item() * imgs.size(0)

                mse, kl = my_evidential_classification(alpha, labels)
                acc["mse_in"]  += mse[regs].sum()
                acc["kl_in"]   += kl[regs].sum()
                acc["mse_out"] += mse[~regs].sum()
                acc["kl_out"]  += kl[~regs].sum()
                acc["loss_in"]  += mse[regs].sum()      + lamb * kl[regs].sum()
                acc["loss_out"] += mse[~regs].sum()    + lamb * kl[~regs].sum()

        # テストログ計算
        n_test = len(test_loader.dataset)
        test_logs["loss"].append(sum_loss / n_test)
        for key in acc:
            test_logs[key].append(acc[key].item() / n_test)

        # CSV 出力
        df = pd.DataFrame({
            "epoch": list(range(1, epoch + 2)),
            **{f"train_{k}": v for k, v in train_logs.items()},
            **{f"test_{k}":  v for k, v in test_logs.items()},
        })
        df.to_csv(os.path.join(result_dir, "loss_status.csv"), index=False)

    # モデル保存
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))

# 実行例
if __name__ == "__main__":
    train(cuda=0, transforms=get_transforms(), save_dir="test")

