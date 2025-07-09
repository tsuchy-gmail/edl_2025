import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def get_case_list(csv_path):
    testdata_df = pd.read_csv(csv_path)
    return testdata_df["case"].tolist()

def get_npz_path(dirname):
    paths = glob.glob(f"{dirname}/alpha/epoch100/values/all_alphas.npz")
    if len(paths) == 0:
        epoch100_paths = glob.glob(f"{dirname}/alpha/*/epoch100*")
        epoch100_dir = os.path.dirname(epoch100_paths[0])
        return os.path.join(epoch100_dir, "values/all_alphas.npz")
    else:
        return paths[0]


def get_alphas_by_model(target_paths):
    alphas_by_model = []
    for target_path in target_paths:
        csv_df = pd.read_csv(target_path)
        if len(csv_df) < 100:
            continue
        dirname = os.path.dirname(target_path)
        npz_path = get_npz_path(dirname)
        alpha_dict = np.load(npz_path)
        alphas_by_model.append(alpha_dict)

    return alphas_by_model



def get_med_std(alphas_by_model, typ, idx):
    alphas_list = []
    for alpha_dict in alphas_by_model:
        target_alphas = alpha_dict[typ]
        alphas_list.append(target_alphas[idx])

    alphas_np = np.array(alphas_list)
    a1 = alphas_np[:, 0]
    a2 = alphas_np[:, 1]
    med1, std1 = np.median(a1), np.std(a1)
    med2, std2 = np.median(a2), np.std(a2)
    std_sum = std1 + std2

    return (med1, med2), std_sum

def create_case_dict(dict_by_case):
    dict_R = {}
    dict_F = {}
    for case, typ_dict in dict_by_case.items():
        for typ, med_std_list in typ_dict.items():
            if "R" in typ:
                if "in" in typ:
                    dict_R[case] = {"in": med_std_list}
                elif "out" in typ:
                    dict_R[case]["out"] = med_std_list
                else:
                    raise ValueError("typに予期せぬ値1")
            elif "F" in typ:
                if "in" in typ:
                    dict_F[case] = {"in": med_std_list}
                elif "out" in typ:
                    dict_F[case]["out"] = med_std_list
                else:
                    raise ValueError("typに予期せぬ値1")
            else:
                raise ValueError("typに予期せぬ値2")

    return dict_R, dict_F


def alpha_scatter(med_std_dict, N, crop_size, lamb, n_model, date, lim, alpha):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    types = ["R_in", "R_out", "F_in", "F_out"]
    for ax, typ in zip(axes.flat, types):
        med, std = med_std_dict[typ]["med"], med_std_dict[typ]["std"]
        a1, a2 = zip(*med)
        std = np.array(std)
        var = std ** 2
        ax.scatter(a1, a2, s=var/10, alpha=alpha)
        ax.set_xlabel("alpha1 - Class:Reactive")
        ax.set_ylabel("alpha2 - Class:FL")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{typ}, n={N}, mean:{np.mean(var/10):.3f}, std:{np.std(var/10):.3f}")
        print("sum(var)",np.sum(var))
        print("var.shape", var.shape)

    fig.suptitle(f"size:{crop_size}, lamb:{lamb}, n_model:{n_model}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.tight_layout()
    save_dir = f"result/2025/inter_model_var/{date}/lim{lim}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"var_circle_lamb{lamb}_{crop_size}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

def create_med_std_dict(alphas_by_model, N):
    subtypes = ["R", "F"]
    regions = ["in", "out"]
    med_std_dict = {}
    for region in regions:
        for subtype in subtypes:
            med_list = []
            std_list = []
            typ = f"{subtype}_{region}"
            for i in range(N):
                med, std = get_med_std(alphas_by_model, typ, i)
                med_list.append(med)
                std_list.append(std)
            med_std_dict[typ] = {"med": med_list, "std": std_list}
    
    return med_std_dict

def create_dict_by_case(alphas_by_model, N, case_list):
    subtypes = ["R", "F"]
    regions = ["in", "out"]
    alpha_dict_by_case = defaultdict(lambda: defaultdict(list))
    for i, region in enumerate(regions):
        for j, subtype in enumerate(subtypes):
            typ = f"{subtype}_{region}"
            for k in range(N):
                med, std = get_med_std(alphas_by_model, typ, k)
                case = case_list[2*N*i + N*j + k]
                alpha_dict_by_case[case][typ].append([med[0],med[1],std])

    return alpha_dict_by_case

def scatter_by_case(subtype_dict, subtype, crop_size, lamb, n_model, date, lim, alpha):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    for i, (case, region_d) in enumerate(subtype_dict.items()):
        for j, region in enumerate(["in", "out"]):
            med_std_list = region_d[region]
            ax = axes[i,j]
            med1, med2, std = zip(*med_std_list)
            std = np.array(std)
            var = std ** 2
            ax.scatter(med1, med2, s=var/10, alpha=alpha)
            ax.set_xlabel("alpha1 - Class:Reactive")
            ax.set_ylabel("alpha2 - Class:FL")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_title(f"{case}_{region}, mean:{np.mean(var/10):.3f}, std:{np.std(var/10):.3f}")

    fig.suptitle(f"subtype:{subtype}\nsize:{crop_size}, lamb:{lamb}, n_model:{n_model}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.tight_layout()
    save_dir = f"result/2025/inter_model_var/{date}/lim{lim}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"var_{subtype}_by_case_lamb{lamb}_{crop_size}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()




def main():
    N = 934
    crop_size = 224

    sizes = [224, 448]
    lambs = ["0to1", "1to0"]
    date = "07_01"
    for lamb in lambs:
        root = f"result/2025/{date}/crop{crop_size}/lamb{lamb}/*/train_loss.csv"
        csv_path = "csv/test_data.csv"
        target_paths = glob.glob(root)
        case_list = get_case_list(csv_path)

        alphas_by_model = get_alphas_by_model(target_paths)
        n_model = len(alphas_by_model)
        print(n_model, lamb)
        med_std_dict = create_med_std_dict(alphas_by_model, N)
        dict_by_case = create_dict_by_case(alphas_by_model, N, case_list)
        dict_R, dict_F = create_case_dict(dict_by_case)

        alpha = 0.85
        for lim in [30, 50, 100]:
            alpha_scatter(med_std_dict, N, crop_size, lamb, n_model, date, lim, alpha)
            scatter_by_case(dict_R, "R", crop_size, lamb, n_model, date, lim, alpha)
            scatter_by_case(dict_F, "F", crop_size, lamb, n_model, date, lim, alpha)


    #alpha_scatter(med_std_dict, N)
main()



