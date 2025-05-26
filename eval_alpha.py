import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


subtypes = ["R", "F"]
regions = ["in", "out"]

root = "./result/"
target_dirs = ["05_21", "05_23"]


def get_p_means():
    p_means = {
            "R_in": [],
            "R_out": [],
            "F_in": [],
            "F_out": [],
            }
    for subtype in subtypes:
        for region in regions:
            filename = f"{subtype}_{region}.csv"
            for target_dir in target_dirs:
                model_root_dir = os.path.join(root, target_dir)
                model_dirs = os.listdir(model_root_dir)
                for model_dir in model_dirs:
                    alpha_dir = os.path.join(model_root_dir, model_dir, "alpha_last")
                    if os.path.exists(alpha_dir):
                        csv_path = os.path.join(alpha_dir, filename)
                        alpha_df = pd.read_csv(csv_path)
                        p_df = alpha_df.div(alpha_df.sum(axis=1), axis=0)
                        p_mean_R = p_df["0"].mean()
                        p_mean_F = p_df["1"].mean()

                        p_means[f"{subtype}_{region}"].append((p_mean_R, p_mean_F))
    
    return p_means

def get_alpha_means():
    alpha_means = {
            "R_in": [],
            "R_out": [],
            "F_in": [],
            "F_out": [],
            }
    for subtype in subtypes:
        for region in regions:
            filename = f"{subtype}_{region}.csv"
            for target_dir in target_dirs:
                model_root_dir = os.path.join(root, target_dir)
                model_dirs = os.listdir(model_root_dir)
                for model_dir in model_dirs:
                    alpha_dir = os.path.join(model_root_dir, model_dir, "alpha_last")
                    if os.path.exists(alpha_dir):
                        csv_path = os.path.join(alpha_dir, filename)
                        alpha_df = pd.read_csv(csv_path)

                        alpha_mean_R = alpha_df["0"].mean()
                        alpha_mean_F = alpha_df["1"].mean()

                        alpha_means[f"{subtype}_{region}"].append((alpha_mean_R, alpha_mean_F))
    
    return alpha_means


#pd.DataFrame(alpha_means).to_csv("result/eval_alpha.csv")

def hist(data, subtype, region, label, bins):
    mean = np.mean(data)
    var = np.var(data)
    plt.hist(data, bins=bins, color='skyblue', density=True, edgecolor='black')
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title(f"input:{subtype}-{region}\nmean:{mean:.2f}\nvar:{var:.2f}")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"result/{subtype}_{region}_{label}.png")
    plt.close()




def fig(means, name1, name2):
    for subtype in subtypes:
        for region in regions:
            vec_list = means[f"{subtype}_{region}"]

            a1_list = [a1 for a1, a2 in vec_list]
            a2_list = [a2 for a1, a2 in vec_list]

            hist(a1_list, subtype, region, name1, 5)
            hist(a2_list, subtype, region, name2, 5)


p_means = get_p_means()
fig(p_means, "p1", "p2")
#alpha_means = get_alpha_means()
#fig(alpha_means)
