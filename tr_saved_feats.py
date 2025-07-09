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
from collections import defaultdict
from torchvision.transforms.v2 import ColorJitter, RandomApply

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torch.amp import autocast


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

class FeatureDataset(Dataset):
    def __init__(self, features, subtypes, regions, cases):
        self.features = features
        self.subtypes = subtypes
        self.regions = regions
        self.cases = cases
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx].copy())
        label = encode_subtype(self.subtypes[idx])
        region = encode_region(self.regions[idx])
        case = self.cases[idx]
        return feat, label, region, case

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
        region = self.region_list[idx]
        region = encode_region(region)
        case = self.case_list[idx]

        return img, label, region, case

def get_list_data(csv_path):
    data_df = read_csv(csv_path)

    img_path_list = data_df["img_path"].tolist()
    subtype_list = data_df["subtype"].tolist()
    region_list = data_df["region"].tolist()
    case_list = data_df["case"].tolist()

    return img_path_list, subtype_list, region_list, case_list


class RandomRotation90:
    def __call__(self, img):
        angles = [90, 180, 270]
        angle = random.choice(angles)

        return transforms.functional.rotate(img, angle)

def get_transforms(crop_size):
    crop_size_tuple = (crop_size, crop_size)

    return transforms.Compose(
            [
                #RandomRotation90(),
                transforms.CenterCrop(size=crop_size_tuple),
                #transforms.ToTensor(),
                transforms.PILToTensor(),
            ]
    )

def to_avg_dict(epoch, loss, loss_in, loss_out, mse_in, mse_out, kl_in, kl_out, n_data, lamb, acc=None, acc_in=None, acc_out=None):
    n_data_half = n_data // 2
    if acc and acc_in and acc_out:
        return {
                "epoch": epoch + 1,
                "lambda": lamb,
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
                "lambda": lamb,
                "loss": loss / n_data,
                "loss_in": loss_in / n_data_half,
                "loss_out": loss_out / n_data_half,
                "mse_in": mse_in / n_data_half,
                "mse_out": mse_out / n_data_half,
                "kl_in": kl_in / n_data_half,
                "kl_out": kl_out / n_data_half,
                }

def scatter(alpha_df, save_root_dir, typ):
    plt.scatter(alpha_df[0], alpha_df[1], s=1, alpha=0.2)
    save_path = os.path.join(save_root_dir, f"scatter_{typ}.png")
    plt.grid(True)
    plt.title(f"alpha\ninput:{typ}")
    plt.xlabel("alpha1(class: Reactive)")
    plt.ylabel("alpha2(class: FL)")
    plt.savefig(save_path, dpi=300)
    plt.close()

def scatter_all_types(alpha_dict, save_dir, is_abs=False):
    fig, axes = plt.subplots(2, 2, figsize=(6,6))
    types = ["R_in", "R_out", "F_in", "F_out"]
    for ax, typ in zip(axes.flat, types):
        target_alpha = alpha_dict[typ]
        a1, a2 = target_alpha[:, 0], target_alpha[:, 1]
        
        n_all = target_alpha.shape[0]
        n_pred_R = (a1 > a2).sum()
        n_pred_F = (a1 < a2).sum()
        subtype = typ[0]

        if subtype == "R":
            acc = n_pred_R / n_all
        if subtype == "F":
            acc = n_pred_F / n_all

        max_a1 = a1.max()
        max_a2 = a2.max()
        lim = max(max_a1, max_a2)
        ax.scatter(a1, a2, s=1, alpha=0.2)
        ax.set_title(f"{typ}, acc:{acc:.2f}")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        if is_abs:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
    plt.tight_layout()
    
    file_suffix = "_abs" if is_abs else ""
    save_path = os.path.join(save_dir, f"all_cases{file_suffix}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

def scatter_by_case(alpha_dict, save_dir, case_dict, is_abs=False):
    alpha_dict_by_case = defaultdict(lambda: defaultdict(list))
    types = ["R_in", "R_out", "F_in", "F_out"]
    for typ in types:
        target_alpha = alpha_dict[typ]
        target_case = case_dict[typ]
        for alpha, case in zip(target_alpha, target_case):
            alpha_dict_by_case[case][typ].append(alpha)

    
    cases_R = list(set(case_dict["R_in"]))
    cases_F = list(set(case_dict["F_in"]))

    for subtype_idx, cases in enumerate([cases_R, cases_F]):
        subtype = "R" if subtype_idx == 0 else "F"
        fig, axes = plt.subplots(len(cases), 2, figsize=(6, 3 * len(cases)))
        for i, case in enumerate(cases):
            for j, region in enumerate(["in", "out"]):
                ax = axes[i,j]
                typ = f"{subtype}_{region}"
                alpha_list = alpha_dict_by_case[case][typ]
                alpha_np = np.stack(alpha_list)
                a1, a2 = alpha_np[:, 0], alpha_np[:, 1]

                n_all = alpha_np.shape[0]
                n_pred_R = (a1 > a2).sum()
                n_pred_F = (a1 < a2).sum()
                if subtype == "R":
                    acc = n_pred_R / n_all
                if subtype == "F":
                    acc = n_pred_F / n_all

                ax.scatter(a1, a2, s=1, alpha=0.3)

                title = f"{case}_{region}, acc:{acc:.2f}"
                ax.set_title(title)

                max_a1 = a1.max()
                max_a2 = a2.max()
                lim = max(max_a1, max_a2)
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                if is_abs:
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
        fig.suptitle(subtype, fontsize=16)
        plt.tight_layout()
        file_suffix = "_abs" if is_abs else ""
        save_path = os.path.join(save_dir, f"by_case_{subtype}{file_suffix}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

def hist(a, b, title, save_path):
    plt.hist([a,b], label=["correct", "incorrect"], bins=np.arange(0, 1.1, 0.1), color=["orange", "blue"])
    plt.xticks(np.arange(0,1.1,0.1))
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def unc_hist(unc_dict, region, unc_dir):
    subtypes = ["R", "F"]

    corr_list = []
    inco_list = []
    for subtype in subtypes:
        unc_corr = unc_dict[f"pred_{subtype}"]["correct"]
        unc_inco = unc_dict[f"pred_{subtype}"]["incorrect"]
        corr_list.append(unc_corr)
        inco_list.append(unc_inco)
        full_label = "Reactive" if subtype == "R" else "FL"
        title = f"Histgram (prediction: {full_label}, region:{region})"
        save_path = os.path.join(unc_dir, f"histgram_{subtype}_{region}.png")
        hist(unc_corr, unc_inco, title, save_path)

    unc_corr_all = np.concatenate(corr_list)
    unc_inco_all = np.concatenate(inco_list)
    title = f"Histgram (all prediction, region:{region})"
    save_path = os.path.join(unc_dir, f"histgram_{region}_all.png")
    hist(unc_corr_all, unc_inco_all, title, save_path)


def confmx(alpha_dict, alpha_root_dir, region):
    subtypes = ["R", "F"]
    count_row_R = []
    count_row_F = []

    unc_row_R = []
    unc_row_F = []
    n_all = 0
    correct = 0

    acc_R = 0
    acc_F = 0

    unc_dict = {}
    for subtype in subtypes:
        alpha = alpha_dict[f"{subtype}_{region}"]
        n_all += alpha.shape[0]

        mask_pred_R = alpha[:, 0] > alpha[:, 1]
        mask_pred_F = alpha[:, 0] < alpha[:, 1]

        count_R = np.sum(mask_pred_R)
        count_F = np.sum(mask_pred_F)

        alpha_R = alpha[mask_pred_R]
        alpha_F = alpha[mask_pred_F]

        
        unc_R = alpha.shape[1] / (alpha_R[:,0] + alpha_R[:,1])
        unc_F = alpha.shape[1] / (alpha_F[:,0] + alpha_F[:,1])

        unc_mean_R = np.mean(unc_R)
        unc_mean_F = np.mean(unc_F)

        unc = alpha.shape[1] / (alpha[:, 0] + alpha[:, 1])
        
        if subtype == "R":
            count_row_R.extend([count_R, count_F])
            unc_row_R.extend([unc_mean_R, unc_mean_F])
            correct += count_R
            acc_R = count_R / alpha.shape[0]

            unc_dict["pred_R"] = {"correct": unc_R, "incorrect": unc_F}
        elif subtype == "F":
            count_row_F.extend([count_R, count_F])
            unc_row_F.extend([unc_mean_R, unc_mean_F])
            correct += count_F
            acc_F = count_F / alpha.shape[0]

            unc_dict["pred_F"] = {"correct": unc_F, "incorrect": unc_R}
        else:
            raise ValueError("subtypeに予期せぬ値")
    
    confmx_dir = os.path.join(alpha_root_dir, "confusion_matrix")
    unc_dir = os.path.join(alpha_root_dir, "uncertainty")
    os.makedirs(confmx_dir, exist_ok=True)
    os.makedirs(unc_dir, exist_ok=True)

    count_col_R, count_col_F = map(list, zip(count_row_R, count_row_F))
    confmx_dict = {"R": count_col_R, "F": count_col_F}
    confmx_save_path = os.path.join(confmx_dir, f"confmx_{region}.csv")
    confmx_df = pd.DataFrame(confmx_dict, index=subtypes)
    confmx_df.to_csv(confmx_save_path, index=True, index_label="true")

    unc_col_R, unc_col_F = map(list, zip(unc_row_R, unc_row_F))
    uncmx_dict = {"R": unc_col_R, "F": unc_col_F}
    uncmx_save_path = os.path.join(unc_dir, f"uncmx_{region}.csv")
    uncmx_df = pd.DataFrame(uncmx_dict, index=subtypes)
    uncmx_df.to_csv(uncmx_save_path, index=True, index_label="true")

    unc_hist(unc_dict, region, unc_dir)

    acc_save_path = os.path.join(confmx_dir, f"acc_{region}.txt")
    acc = correct / n_all

    with open(acc_save_path, "w") as f:
        f.write(f"accuracy:{acc:.2f}\n")
        f.write(f"accuracy_R:{acc_R:.2f}\n")
        f.write(f"accuracy_F:{acc_F:.2f}\n")

def plot_loss(loss_df, title, save_path, lim):
    loss_df.plot(x="epoch", y=["loss", "loss_in", "loss_out"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, lim)
    plt.yticks(np.arange(0, lim + 0.2, 0.2))
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_acc(loss_df, title, save_path):
    loss_df.plot(x="epoch", y=["acc", "acc_in", "acc_out"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def test(head, loader, records, save_ok, device, lamb, epoch, result_path, epochs, min_test_loss_in, max_acc_in):
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

    alpha_list_dict = {
            "R_in": [],
            "R_out": [],
            "F_in": [],
            "F_out": [],
            }
    case_list_dict = {
            "R_in": [],
            "R_out": [],
            "F_in": [],
            "F_out": [],
            }
    subtypes = ["R", "F"]
    regions = ["in", "out"]

    with torch.no_grad():
        head.eval()
        for feat_batch, label_batch, region_batch, case_batch in loader:
            feat_batch = feat_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            region_batch = region_batch.to(device, non_blocking=True)
            case_batch = np.array(case_batch)

            alpha_batch = head(feat_batch)

            loss = evidential_classification(alpha_batch, label_batch, lamb)
            loss_total += loss.item() * feat_batch.size(0)
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            R_mask = label_batch == R
            F_mask = label_batch == F
            in_mask = region_batch == IN
            out_mask = region_batch == OUT

            R_in_mask = R_mask & in_mask
            R_out_mask = R_mask & out_mask
            F_in_mask = F_mask & in_mask
            F_out_mask = F_mask & out_mask

            mask = {
                "R_in": R_in_mask,
                "R_out": R_out_mask,
                "F_in": F_in_mask,
                "F_out": F_out_mask,
                    }
            for subtype in subtypes:
                for region in regions:
                    typ = f"{subtype}_{region}"
                    target_alpha = alpha_batch[mask[typ]]
                    target_case = case_batch[mask[typ].cpu()] #labelとかGPUに送ってるからmaskもGPU上にある
                    alpha_list_dict[typ].append(target_alpha)
                    case_list_dict[typ].append(target_case)

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
    
    n_data = len(loader.dataset)
    n_data_in = n_data_in.item()
    n_data_out = n_data_out.item()

    acc = (n_correct.item() / n_data)
    acc_in = (n_correct_in.item() / n_data_in)
    acc_out = (n_correct_out.item() / n_data_out)


    avg_dict = to_avg_dict(epoch, loss_total, loss_in_total, loss_out_total, mse_in_total, mse_out_total, kl_in_total, kl_out_total, n_data, lamb, acc, acc_in, acc_out)
    records.append(avg_dict)

    avg_loss_in = avg_dict["loss_in"]
    is_min_loss_in = avg_loss_in < min_test_loss_in
    is_max_acc_in  = acc_in > max_acc_in

    if is_min_loss_in:
        min_test_loss_in = avg_loss_in
    if is_max_acc_in:
        max_acc_in = acc_in

    alpha_save_cond = (epoch+1) % 10 == 0 or epoch == 0 or is_min_loss_in or is_max_acc_in

    if alpha_save_cond:
        alpha_dirname = ""
        if is_min_loss_in or is_max_acc_in:
            if is_min_loss_in:
                alpha_dirname = "min_testloss_in"
            if is_max_acc_in:
                alpha_dirname = "max_acc_in"
            if is_min_loss_in and is_max_acc_in:
                alpha_dirname = "minloss_maxacc_in"
        else:
           alpha_dirname = f"epoch{epoch+1}"



        alpha_root_dir = os.path.join(result_path, "alpha", f"{alpha_dirname}")
        os.makedirs(alpha_root_dir, exist_ok=True)
        if is_min_loss_in or is_max_acc_in:
            epoch_memo_path = os.path.join(alpha_root_dir, f"epoch{epoch+1}.txt")
            open(epoch_memo_path, "w").close()

        alpha_values_dir = os.path.join(alpha_root_dir, "values")
        alpha_scatter_dir = os.path.join(alpha_root_dir, "scatter")
        os.makedirs(alpha_values_dir, exist_ok=True)
        os.makedirs(alpha_scatter_dir, exist_ok=True)

        alpha_dict = {
                "R_in": [],
                "R_out": [],
                "F_in": [],
                "F_out": [],
                }
        case_dict = {
                "R_in": [],
                "R_out": [],
                "F_in": [],
                "F_out": [],
                }
        for subtype in subtypes:
            for region in regions:
                typ = f"{subtype}_{region}"
                alpha_dict[typ] = torch.cat(alpha_list_dict[typ], dim=0).cpu().numpy()
                case_dict[typ] = np.concatenate(case_list_dict[typ], axis=0)

        scatter_all_types(alpha_dict, alpha_scatter_dir)
        scatter_all_types(alpha_dict, alpha_scatter_dir, True)
        scatter_by_case(alpha_dict, alpha_scatter_dir, case_dict)
        scatter_by_case(alpha_dict, alpha_scatter_dir, case_dict, True)

                #alpha_df = pd.DataFrame(alpha_dict[typ])
                #case_df = pd.DataFrame(case_dict[typ])
                #alpha_csv_path = os.path.join(alpha_values_dir, f"{typ}.csv")
                #alpha_df.to_csv(alpha_csv_path, index=False)
                #scatter(alpha_df, alpha_scatter_dir, typ)
                #scatter(case_df, alpha_scatter_dir, typ)

        for region in regions:
            confmx(alpha_dict, alpha_root_dir, region)
            
        npz_path = os.path.join(alpha_values_dir, "all_alphas.npz")
        np.savez(npz_path, R_in=alpha_dict["R_in"], R_out=alpha_dict["R_out"], F_in=alpha_dict["F_in"], F_out=alpha_dict["F_out"])

    if save_ok:
        csv_save_path = os.path.join(result_path, "test_loss.csv")
        loss_png_path = os.path.join(result_path, "test_loss.png")
        acc_png_path = os.path.join(result_path, "acc.png")

        test_loss_df = pd.DataFrame(records)
        test_loss_df.to_csv(csv_save_path, index=False)
        plot_loss(test_loss_df, "Test Loss", loss_png_path, 2)
        plot_acc(test_loss_df, "Accuracy", acc_png_path)

        #torch.save(model.state_dict(), os.path.join(result_path, "model_last.pth"))

    return avg_dict["loss_in"], avg_dict["loss_out"], acc, acc_in, acc_out

def get_loader(ini_num_workers, num_workers, batch_size, crop_size):
    _, train_subtype_list, train_region_list, train_case_list= get_list_data("csv/train_data.csv")
    train_feats = np.load("vir_feats/resize224_train.npy", mmap_mode="r")
    train_ds = FeatureDataset(train_feats, train_subtype_list, train_region_list, train_case_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    _, test_subtype_list, test_region_list, test_case_list = get_list_data("csv/test_data.csv")
    test_feats = np.load("vir_feats/resize224_test.npy", mmap_mode="r")
    test_ds = FeatureDataset(test_feats, test_subtype_list, test_region_list, test_case_list)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader

def save_imgs_sample(imgs, cases, labels, regions, save_dir):
    imgs = imgs.cpu().numpy() 
    imgs = imgs.transpose(0, 2, 3, 1)
    row, col = 5, 5

    plt.figure(figsize=(20, 20))

    for i in range(len(imgs)):
        plt.subplot(row, col, i+1)
        plt.imshow(imgs[i])
        label = "R" if labels[i] == R else "F"
        region = "out" if regions[i] == OUT else "in"
        case = cases[i]
        plt.title(f"{case}, {label}, {region}")
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "train_imgs_sample.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


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

    device = torch.device(f"cuda:{cuda}")  

    print("preloading start")
    train_records = []

    head = Dirichlet(2560, 2).to(device) #あえて明示
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    #model.fc = Dirichlet(n_features, n_classes)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_records = []
    min_test_loss_in = float("inf")
    min_test_loss_out = float("inf")
    max_acc = -1
    max_acc_in = -1
    max_acc_out = -1

    for epoch in range(epochs):
        print(f"epoch{epoch + 1}")
        head.train()

        loss_total = 0.0
        loss_in_total = 0.0
        loss_out_total = 0.0
        mse_in_total = 0.0
        mse_out_total = 0.0
        kl_in_total = 0.0
        kl_out_total = 0.0

        if "lamb0to1" in lamb_type:
            #lamb = min(1, epoch / 10)
            lamb = epoch / epochs
            #if epoch >= 1000:
            #    lamb = 1
        elif "lamb1to0" in lamb_type:
            lamb = 1 - epoch / epochs
            #lamb = max(0, 1 - epoch / 10)
            #if epoch >= 1000:
            #    lamb = 0.001


        
        for feat_batch, label_batch, region_batch, case_batch in train_loader:
            feat_batch = feat_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            region_batch = region_batch.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16): 
                alpha_batch = head(feat_batch)
                loss = evidential_classification(alpha_batch, label_batch, lamb)

            loss_total += loss.item() * feat_batch.size(0) #バッチ平均 * バッチサイズ = バッチの合計ロス
            mse_batch, kl_batch = my_evidential_classification(alpha_batch, label_batch)

            in_mask = region_batch == IN
            out_mask = region_batch == OUT

            mse_in_total += mse_batch[in_mask].sum().item()
            mse_out_total += mse_batch[out_mask].sum().item()
            kl_in_total += kl_batch[in_mask].sum().item()
            kl_out_total += kl_batch[out_mask].sum().item()

            kl_batch_in = kl_batch[in_mask]
            kl_in_mt1 = kl_batch_in[kl_batch_in >= 1]
            alpha_batch_in = alpha_batch[in_mask]
            alpha_in_mt1 = alpha_batch_in[kl_batch_in >= 1]
            pred_in_batch = torch.argmax(alpha_batch_in, dim=1)
            label_in_batch = label_batch[in_mask]
            correct_in = (pred_in_batch == label_in_batch).sum()
            alpha_inco = alpha_batch_in[pred_in_batch != label_in_batch]
            loss_in_total += (mse_batch[in_mask] + lamb * kl_batch[in_mask]).sum().item()
            loss_out_total += (mse_batch[out_mask] + lamb * kl_batch[out_mask]).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        n_data = len(train_loader.dataset)

        if save_ok:
            avg_dict = to_avg_dict(epoch, loss_total, loss_in_total, loss_out_total, mse_in_total, mse_out_total, kl_in_total, kl_out_total, n_data, lamb)
            train_records.append(avg_dict)
            csv_save_path = os.path.join(result_dir_path, "train_loss.csv")
            loss_png_path = os.path.join(result_dir_path, "train_loss.png")

            train_loss_df = pd.DataFrame(train_records)
            train_loss_df.to_csv(csv_save_path, index=False)
            plot_loss(train_loss_df, "Train Loss", loss_png_path, 2)


        avg_test_loss_in, avg_test_loss_out, acc, acc_in, acc_out = test(head, test_loader, test_records, save_ok, device, lamb, epoch, result_dir_path, epochs, min_test_loss_in, max_acc_in)

        if avg_test_loss_in < min_test_loss_in:
            min_test_loss_in = avg_test_loss_in
        """
            torch.save(head.state_dict(), os.path.join(result_dir_path, "min_test_loss_in.pth"))
        if avg_test_loss_out < min_test_loss_out:
            min_test_loss_out = avg_test_loss_out
            torch.save(model.state_dict(), os.path.join(result_dir_path, "min_test_loss_out.pth"))

        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), os.path.join(result_dir_path, "max_acc.pth"))
        """
        if acc_in > max_acc_in:
            max_acc_in = acc_in
        """
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
    crop_size = int(sys.argv[6+n_ps]) if len(sys.argv) >= (7+n_ps) else 224

    #batch_size = 64 * ((448 // crop_size) ** 2)
    batch_size = 256

    train_loader, test_loader = get_loader(ini_num_workers, num_workers, batch_size, crop_size)


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
