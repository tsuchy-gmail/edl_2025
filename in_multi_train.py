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
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import ColorJitter, RandomApply

# =========================
# 定数
# =========================
REACTIVE = "Reactive"
FL = "FL"
R = 0
F = 1
INSIDE = "inside"
OUTSIDE = "outside"
OUT = 0
IN = 1
n_classes = 2
learning_rate = 1e-4
cpu_count = os.cpu_count()

# =========================
# ラベルエンコード
# =========================
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

# =========================
# 前ロード用 Dataset
# =========================
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

def preload_all_imgs(img_path_list, transform, ini_num_workers, batch_size, crop_size):
    img_ds = ImageDataset(img_path_list, transform)
    img_loader = DataLoader(img_ds, batch_size=batch_size, shuffle=False, num_workers=ini_num_workers)
    n_all_imgs = len(img_path_list)
    all_imgs_tensor = torch.empty((n_all_imgs, 3, 224, 224), dtype=torch.uint8)

    start = 0
    for i, img_batch in enumerate(tqdm(img_loader, desc="Loading imgs in batch")):
        end = start + img_batch.size(0)
        all_imgs_tensor[start:end].copy_(img_batch)
        start = end

    return all_imgs_tensor

# =========================
# 学習用 Dataset（互換維持のため region も返すが inside 固定）
# =========================
class CustomDataset(Dataset):
    def __init__(self, imgs_tensor, subtype_list, region_list, case_list):
        self.imgs_tensor = imgs_tensor
        self.subtype_list = subtype_list
        self.region_list = region_list
        self.case_list = case_list

    def __len__(self):
        return self.imgs_tensor.size(0)

    def __getitem__(self, idx):
        img = self.imgs_tensor[idx]
        subtype = self.subtype_list[idx]
        label = encode_subtype(subtype)
        region = encode_region(INSIDE)  # inside のみ
        case = self.case_list[idx]
        return img, label, region, case

# =========================
# CSV -> リスト化（inside のみ）
# =========================
def get_list_data(csv_path):
    df = read_csv(csv_path)
    df = df[df["region"] == INSIDE].reset_index(drop=True)

    img_path_list = df["img_path"].tolist()
    subtype_list = df["subtype"].tolist()
    region_list = df["region"].tolist()
    case_list = df["case"].tolist()

    return img_path_list, subtype_list, region_list, case_list

# =========================
# 変換
# =========================
class RandomRotation90:
    def __call__(self, img):
        angles = [90, 180, 270]
        angle = random.choice(angles)
        return transforms.functional.rotate(img, angle)

def get_transforms(crop_size):
    crop_size_tuple = (crop_size, crop_size)

    return transforms.Compose([
        RandomRotation90(),
        transforms.CenterCrop(size=crop_size_tuple),
        transforms.Resize((224, 224)),
        transforms.PILToTensor(),  # 0-255 (uint8)
    ])

# =========================
# 可視化ユーティリティ（insideのみ）
# =========================
def scatter_inside(alpha_dict, save_dir, is_abs=False):
    # 2面（R_in, F_in）
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    types = ["R_in", "F_in"]
    for ax, typ in zip(axes.flat, types):
        if len(alpha_dict[typ]) == 0:
            ax.set_title(f"{typ} (no data)")
            ax.axis("off")
            continue
        a = alpha_dict[typ]
        a1, a2 = a[:,0], a[:,1]
        n_all = a.shape[0]
        n_pred_R = (a1 > a2).sum()
        n_pred_F = (a1 < a2).sum()
        acc = (n_pred_R / n_all) if typ.startswith("R") else (n_pred_F / n_all)
        ax.scatter(a1, a2, s=1, alpha=0.25)
        ax.set_title(f"{typ}, acc:{acc:.2f}")
        lim = max(a1.max(), a2.max())
        ax.set_xlim(0, 100 if is_abs else lim)
        ax.set_ylim(0, 100 if is_abs else lim)
        ax.set_xlabel("alpha1 (Reactive)")
        ax.set_ylabel("alpha2 (FL)")
        ax.grid(True)
    plt.tight_layout()
    suffix = "_abs" if is_abs else ""
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"all_cases{suffix}.png"), dpi=200)
    plt.close()

def scatter_by_case_inside(alpha_dict, case_dict, save_dir, is_abs=False):
    # subtypeごとに症例別2列（inのみ）
    for subtype in ["R", "F"]:
        cases = sorted(list(set(case_dict[f"{subtype}_in"])))
        if len(cases) == 0:
            continue
        fig, axes = plt.subplots(len(cases), 1, figsize=(6, 3*len(cases)))
        if len(cases) == 1:
            axes = [axes]
        for i, case in enumerate(cases):
            # 集約
            mask = np.array(case_dict[f"{subtype}_in"]) == case
            a = alpha_dict[f"{subtype}_in"][mask]
            a1, a2 = a[:,0], a[:,1]
            n_all = a.shape[0]
            if n_all == 0:
                axes[i].set_title(f"{case} (no data)")
                axes[i].axis("off")
                continue
            n_pred_R = (a1 > a2).sum()
            n_pred_F = (a1 < a2).sum()
            acc = (n_pred_R / n_all) if subtype == "R" else (n_pred_F / n_all)
            axes[i].scatter(a1, a2, s=1, alpha=0.3)
            axes[i].set_title(f"{case}_in, acc:{acc:.2f}")
            lim = max(a1.max(), a2.max())
            axes[i].set_xlim(0, 100 if is_abs else lim)
            axes[i].set_ylim(0, 100 if is_abs else lim)
            axes[i].grid(True)
            axes[i].set_xlabel("alpha1 (Reactive)")
            axes[i].set_ylabel("alpha2 (FL)")
        fig.suptitle(subtype, fontsize=14)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        suffix = "_abs" if is_abs else ""
        plt.savefig(os.path.join(save_dir, f"by_case_{subtype}{suffix}.png"), dpi=200)
        plt.close()

def _hist(a, b, title, save_path):
    plt.hist([a, b], label=["correct", "incorrect"], bins=np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(True); plt.title(title); plt.xlabel("Uncertainty"); plt.ylabel("Frequency"); plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()

def unc_hist_inside(unc_dict, save_root_dir):
    # subtype 別 + all（inのみ）
    corr_list, inco_list = [], []
    for subtype in ["R", "F"]:
        unc_corr = unc_dict[f"pred_{subtype}"]["correct"]
        unc_inco = unc_dict[f"pred_{subtype}"]["incorrect"]
        corr_list.append(unc_corr); inco_list.append(unc_inco)
        title = f"Histogram (prediction: {'Reactive' if subtype=='R' else 'FL'}, region: in)"
        save_path = os.path.join(save_root_dir, f"histgram_{subtype}_in.png")
        _hist(unc_corr, unc_inco, title, save_path)
    unc_corr_all = np.concatenate(corr_list) if corr_list else np.array([])
    unc_inco_all = np.concatenate(inco_list) if inco_list else np.array([])
    if len(unc_corr_all) and len(unc_inco_all):
        _hist(unc_corr_all, unc_inco_all, "Histogram (all prediction, region: in)",
              os.path.join(save_root_dir, "histgram_in_all.png"))

def confmx_inside(alpha_dict, alpha_root_dir):
    # 2x2 混同行列（inのみ）
    subtypes = ["R", "F"]
    count_row_R, count_row_F = [], []
    unc_row_R, unc_row_F = [], []
    n_all = 0; correct = 0; acc_R = 0; acc_F = 0
    unc_dict = {}

    for subtype in subtypes:
        alpha = alpha_dict[f"{subtype}_in"]
        if alpha.shape[0] == 0:
            continue
        n_all += alpha.shape[0]
        mask_pred_R = alpha[:,0] > alpha[:,1]
        mask_pred_F = alpha[:,0] < alpha[:,1]
        count_R = np.sum(mask_pred_R)
        count_F = np.sum(mask_pred_F)
        alpha_R = alpha[mask_pred_R]
        alpha_F = alpha[mask_pred_F]
        # uncertainty = K / sum(alpha) with K=2
        unc_R = 2.0 / (alpha_R[:,0] + alpha_R[:,1]) if alpha_R.size else np.array([])
        unc_F = 2.0 / (alpha_F[:,0] + alpha_F[:,1]) if alpha_F.size else np.array([])
        unc_mean_R = float(np.mean(unc_R)) if unc_R.size else np.nan
        unc_mean_F = float(np.mean(unc_F)) if unc_F.size else np.nan

        if subtype == "R":
            count_row_R.extend([count_R, count_F])
            unc_row_R.extend([unc_mean_R, unc_mean_F])
            correct += count_R
            acc_R = count_R / alpha.shape[0]
            unc_dict["pred_R"] = {"correct": unc_R, "incorrect": unc_F}
        else:
            count_row_F.extend([count_R, count_F])
            unc_row_F.extend([unc_mean_R, unc_mean_F])
            correct += count_F
            acc_F = count_F / alpha.shape[0]
            unc_dict["pred_F"] = {"correct": unc_F, "incorrect": unc_R}

    confmx_dir = os.path.join(alpha_root_dir, "confusion_matrix")
    unc_dir = os.path.join(alpha_root_dir, "uncertainty")
    os.makedirs(confmx_dir, exist_ok=True); os.makedirs(unc_dir, exist_ok=True)

    if count_row_R and count_row_F:
        count_col_R, count_col_F = map(list, zip(count_row_R, count_row_F))
        confmx_df = pd.DataFrame({"R": count_col_R, "F": count_col_F}, index=subtypes)
        confmx_df.to_csv(os.path.join(confmx_dir, "confmx_in.csv"), index=True, index_label="true")

        uncmx_df = pd.DataFrame({"R": list(map(float, unc_row_R)), "F": list(map(float, unc_row_F))}, index=subtypes)
        uncmx_df.to_csv(os.path.join(unc_dir, "uncmx_in.csv"), index=True, index_label="true")

        acc = correct / n_all if n_all else 0.0
        with open(os.path.join(confmx_dir, "acc_in.txt"), "w") as f:
            f.write(f"accuracy:{acc:.4f}\n")
            f.write(f"accuracy_R:{acc_R:.4f}\n")
            f.write(f"accuracy_F:{acc_F:.4f}\n")

    # histogram
    if "pred_R" in unc_dict and "pred_F" in unc_dict:
        unc_hist_inside(unc_dict, unc_dir)

# =========================
# 記録ユーティリティ
# =========================
def to_avg_dict(epoch, loss, mse_total, kl_total, n_data, lamb, acc=None):
    d = {
        "epoch": epoch + 1,
        "lambda": lamb,
        "loss": loss / n_data,
        "mse": mse_total / n_data,
        "kl": kl_total / n_data,
    }
    if acc is not None:
        d["acc"] = acc
    return d

def plot_loss(loss_df, title, save_path):
    ax = loss_df.plot(x="epoch", y=["loss", "mse", "kl"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value"); ax.set_title(title); ax.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_acc(loss_df, title, save_path):
    if "acc" not in loss_df.columns: return
    ax = loss_df.plot(x="epoch", y=["acc"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0,1); ax.set_title(title); ax.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# =========================
# DataLoader 構築
# =========================
def get_loader(transform, ini_num_workers, num_workers, batch_size, crop_size):
    train_img_path_list, train_subtype_list, train_region_list, train_case_list = get_list_data("csv/size896_stride896/train_data.csv")
    train_img_tensor = preload_all_imgs(train_img_path_list, transform, ini_num_workers, batch_size, crop_size)
    train_ds = CustomDataset(train_img_tensor, train_subtype_list, train_region_list, train_case_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    test_img_path_list, test_subtype_list, test_region_list, test_case_list = get_list_data("csv/size896_stride896/test_data.csv")
    test_img_tensor = preload_all_imgs(test_img_path_list, transform, ini_num_workers, batch_size, crop_size)
    test_ds = CustomDataset(test_img_tensor, test_subtype_list, test_region_list, test_case_list)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader

# =========================
# 画像サンプル保存（任意）
# =========================
def save_imgs_sample(imgs, cases, labels, regions, save_dir):
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)  # CHW -> HWC
    row, col = 5, 5
    plt.figure(figsize=(20, 20))
    for i in range(min(len(imgs), row * col)):
        plt.subplot(row, col, i+1)
        plt.imshow(imgs[i])
        label = "R" if labels[i] == R else "F"
        region = "in"
        case = cases[i]
        plt.title(f"{case}, {label}, {region}")
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "train_imgs_sample.png"), dpi=150)
    plt.close()

# =========================
# テスト（inside のみ＋alpha保存）
# =========================
@torch.no_grad()
def test(model, loader, records, save_ok, device, lamb, epoch, result_path, epochs, min_test_loss, max_acc):
    model.eval()
    loss_total = 0.0
    mse_total = 0.0
    kl_total = 0.0
    n_correct = 0
    n_data = 0

    # alpha収集（insideのみ）
    alpha_list_dict = {"R_in": [], "F_in": []}
    case_list_dict  = {"R_in": [], "F_in": []}

    for img_batch, label_batch, region_batch, case_batch in loader:
        img_batch = img_batch.to(device, non_blocking=True, dtype=torch.float32).div_(255.0)
        label_batch = label_batch.to(device, non_blocking=True)
        case_batch = np.array(case_batch)

        alpha_batch = model(img_batch)

        # 損失
        loss = evidential_classification(alpha_batch, label_batch, lamb)
        mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)
        loss_total += loss.item() * img_batch.size(0)
        mse_total += mse_batch.sum().item()
        kl_total  += kl_batch.sum().item()

        # 予測
        pred = alpha_batch.argmax(-1)
        n_correct += (pred == label_batch).sum().item()
        n_data    += img_batch.size(0)

        # 収集（R_in, F_in）
        R_mask = (label_batch == R).cpu()
        F_mask = (label_batch == F).cpu()
        if R_mask.any():
            alpha_list_dict["R_in"].append(alpha_batch[R_mask].cpu())
            case_list_dict["R_in"].append(case_batch[R_mask.numpy()])
        if F_mask.any():
            alpha_list_dict["F_in"].append(alpha_batch[F_mask].cpu())
            case_list_dict["F_in"].append(case_batch[F_mask.numpy()])

    acc = n_correct / n_data
    avg_dict = to_avg_dict(epoch, loss_total, mse_total, kl_total, n_data, lamb, acc)
    records.append(avg_dict)

    # 保存・描画（loss/acc）
    if save_ok:
        csv_save_path = os.path.join(result_path, "test_loss.csv")
        loss_png_path = os.path.join(result_path, "test_loss.png")
        acc_png_path  = os.path.join(result_path, "acc.png")
        test_loss_df = pd.DataFrame(records)
        test_loss_df.to_csv(csv_save_path, index=False)
        plot_loss(test_loss_df, "Test Loss (inside only)", loss_png_path)
        plot_acc(test_loss_df, "Accuracy (inside only)", acc_png_path)
        torch.save(model.state_dict(), os.path.join(result_path, "model_last.pth"))

    # ---- alpha保存の判定は train() 側で行うので、ここでは値を返す ----
    # ただし、収集した alpha/case は返して可視化側で使う
    # cat/concat して numpy にしておく
    alpha_dict = {"R_in": np.empty((0,2)), "F_in": np.empty((0,2))}
    case_dict  = {"R_in": np.array([]), "F_in": np.array([])}
    for k in ["R_in", "F_in"]:
        if len(alpha_list_dict[k]):
            alpha_dict[k] = torch.cat(alpha_list_dict[k], dim=0).cpu().numpy()
            case_dict[k]  = np.concatenate(case_list_dict[k], axis=0)
    return avg_dict["loss"], acc, alpha_dict, case_dict

# =========================
# 学習（inside のみ）
# =========================
def train(train_loader, test_loader, epochs, cuda, save_ok, dir_suffix, crop_size, lamb_type):
    start_time = time()
    print("training start")
    print(f"epochs: {epochs}")
    if save_ok:
        year = datetime.now().strftime("%Y")
        date = datetime.now().strftime("%m_%d")
        timestamp = datetime.now().strftime("%H%M_%S%f")[:-3]
        result_dir_path = f"result/{year}/{date}/crop{crop_size}/{dir_suffix}/{timestamp}"
        os.makedirs(result_dir_path, exist_ok=True)
    else:
        result_dir_path = "."

    device = torch.device(f"cuda:{cuda}")

    print("model build")
    model = models.resnet18(weights=None)
    n_features = model.fc.in_features
    model.fc = Dirichlet(n_features, n_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.6, hue=0.05)
    color_jitter_random = RandomApply([color_jitter], p=0.8)

    train_records = []
    test_records = []
    min_test_loss = float("inf")
    max_acc = -1.0

    for epoch in range(epochs):
        print(f"epoch{epoch + 1}")
        model.train()

        loss_total = 0.0
        mse_total = 0.0
        kl_total = 0.0

        # lambda スケジュール
        if "lamb0to1" in lamb_type:
            lamb = min(1.0, epoch / 10.0)
        elif "lamb1to0" in lamb_type:
            lamb = max(0.0, 1.0 - epoch / 10.0)
        else:
            lamb = 1.0

        for img_batch, label_batch, region_batch, case_batch in train_loader:
            img_batch = img_batch.to(device, non_blocking=True, dtype=torch.float32).div_(255.0)
            label_batch = label_batch.to(device, non_blocking=True)

            # img_batch = color_jitter_random(img_batch)  # 使うなら有効化（v2前提）

            alpha_batch = model(img_batch)
            loss = evidential_classification(alpha_batch, label_batch, lamb)
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * img_batch.size(0)
            mse_total  += mse_batch.sum().item()
            kl_total   += kl_batch.sum().item()

        n_data = len(train_loader.dataset)
        if save_ok:
            avg_dict_tr = to_avg_dict(epoch, loss_total, mse_total, kl_total, n_data, lamb)
            train_records.append(avg_dict_tr)
            csv_save_path = os.path.join(result_dir_path, "train_loss.csv")
            loss_png_path = os.path.join(result_dir_path, "train_loss.png")
            train_loss_df = pd.DataFrame(train_records)
            train_loss_df.to_csv(csv_save_path, index=False)
            plot_loss(train_loss_df, "Train Loss (inside only)", loss_png_path)

        # ---- テスト + alpha収集 ----
        avg_test_loss, acc, alpha_dict, case_dict = test(
            model, test_loader, test_records, save_ok, device, lamb,
            epoch, result_dir_path, epochs, min_test_loss, max_acc
        )

        # ---- ベスト更新 ----
        is_min_test_loss = avg_test_loss < min_test_loss
        is_max_acc = acc > max_acc
        if is_min_test_loss:
            min_test_loss = avg_test_loss
            if save_ok:
                torch.save(model.state_dict(), os.path.join(result_dir_path, "min_test_loss.pth"))
        if is_max_acc:
            max_acc = acc
            if save_ok:
                torch.save(model.state_dict(), os.path.join(result_dir_path, "max_acc.pth"))

        # ---- alpha 保存判定（10epochごと + 0epoch + ベスト更新時）----
        alpha_save_cond = (epoch == 0) or ((epoch + 1) % 10 == 0) or is_min_test_loss or is_max_acc

        if save_ok and alpha_save_cond:
            if is_min_test_loss and is_max_acc:
                alpha_dirname = "minloss_maxacc_in"
            elif is_min_test_loss:
                alpha_dirname = "min_testloss_in"
            elif is_max_acc:
                alpha_dirname = "max_acc_in"
            else:
                alpha_dirname = f"epoch{epoch+1}"

            alpha_root_dir = os.path.join(result_dir_path, "alpha", alpha_dirname)
            alpha_values_dir = os.path.join(alpha_root_dir, "values")
            alpha_scatter_dir = os.path.join(alpha_root_dir, "scatter")
            os.makedirs(alpha_values_dir, exist_ok=True)
            os.makedirs(alpha_scatter_dir, exist_ok=True)

            # 値保存
            np.savez(
                os.path.join(alpha_values_dir, "all_alphas.npz"),
                R_in=alpha_dict["R_in"],
                F_in=alpha_dict["F_in"],
            )

            # scatter（全体/絶対）
            scatter_inside(alpha_dict, alpha_scatter_dir, is_abs=False)
            scatter_inside(alpha_dict, alpha_scatter_dir, is_abs=True)

            # 症例別 scatter
            scatter_by_case_inside(alpha_dict, case_dict, alpha_scatter_dir, is_abs=False)
            scatter_by_case_inside(alpha_dict, case_dict, alpha_scatter_dir, is_abs=True)

            # confusion & uncertainty（inのみ）
            confmx_inside(alpha_dict, alpha_root_dir)

            # メモ（ベスト更新時）
            if is_min_test_loss or is_max_acc:
                with open(os.path.join(alpha_root_dir, f"epoch{epoch+1}.txt"), "w") as f:
                    f.write("alpha saved on best update\n")

    # ---- 訓練全体時間 ----
    end_time = time()
    elapsed_time = str(timedelta(seconds=(end_time - start_time)))
    print("time", elapsed_time)
    if save_ok:
        with open(os.path.join(result_dir_path, "time.txt"), "w") as f:
            f.write(elapsed_time)

# =========================
# エントリポイント
# =========================
def get_loader(transform, ini_num_workers, num_workers, batch_size, crop_size):
    train_img_path_list, train_subtype_list, train_region_list, train_case_list = get_list_data("csv/size896_stride896/train_data.csv")
    train_img_tensor = preload_all_imgs(train_img_path_list, transform, ini_num_workers, batch_size, crop_size)
    train_ds = CustomDataset(train_img_tensor, train_subtype_list, train_region_list, train_case_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    test_img_path_list, test_subtype_list, test_region_list, test_case_list = get_list_data("csv/size896_stride896/test_data.csv")
    test_img_tensor = preload_all_imgs(test_img_path_list, transform, ini_num_workers, batch_size, crop_size)
    test_ds = CustomDataset(test_img_tensor, test_subtype_list, test_region_list, test_case_list)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader

def main():
    n_ps = int(sys.argv[1])
    cudas = [None] * n_ps
    for i in range(n_ps):
        cudas[i] = int(sys.argv[2+i])
    ini_num_workers = int(sys.argv[2 + n_ps]) if len(sys.argv) >= (3 + n_ps) else cpu_count-1
    num_workers = int(sys.argv[3+n_ps]) if len(sys.argv) >= (4+n_ps) else cpu_count-1
    n_loop = int(sys.argv[4+n_ps]) if len(sys.argv) >= (5+n_ps) else 1
    dir_suffix = sys.argv[5+n_ps] if len(sys.argv) >= (6+n_ps) else ""
    crop_size = int(sys.argv[6+n_ps]) if len(sys.argv) >= (7+n_ps) else 224

    crop_size = 448

    batch_size = 256

    transform = get_transforms(crop_size)
    train_loader, test_loader = get_loader(transform, ini_num_workers, num_workers, batch_size, crop_size)

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
                "crop_size": crop_size,
                "lamb_type": dir_suffix,
            })
            ps.start()
            ps_list.append(ps)

        for ps in ps_list:
            ps.join()

if __name__ == "__main__":
    main()

