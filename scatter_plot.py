import pandas as pd
import matplotlib.pyplot as plt
import os

subtypes = ["R", "F"]
regions = ["in", "out"]

def scatter(target_par_dir):
    model_root_dir = os.path.join("result", target_par_dir)
    model_dir_list = os.listdir(model_root_dir)
    for model_dir in model_dir_list:
        target_model_dir = os.path.join(model_root_dir, model_dir)
        alpha_dir = os.path.join(target_model_dir, "alpha_last")
        if not os.path.exists(alpha_dir):
            continue
        for subtype in subtypes:
            for region in regions:
                filename = f"{subtype}_{region}.csv"
                csv_path = os.path.join(alpha_dir, filename)
                df = pd.read_csv(csv_path)
                plt.scatter(df["0"], df["1"], s=1, alpha=0.2)
                save_path = os.path.join(alpha_dir, f"scatter_{subtype}_{region}.png")
                plt.grid(True)
                plt.title(f"alpha\ninput:{subtype}-{region}")
                plt.xlabel("alpha1(class: Reactive)")
                plt.ylabel("alpha2(class: FL)")
                plt.savefig(save_path, dpi=300)
                print("saved")
                plt.close()

def conf_mx(region):
    n_all = 0
    correct = 0
    for subtype in subtypes:
        filename = f"alpha_{subtype}_{region}.csv"
        csv_path = os.path.join(result_dir, filename)
        df = pd.read_csv(csv_path)
        R_count = (df["0"] > df["1"]).sum()
        F_count = (df["0"] < df["1"]).sum()
        eq_count = (df["0"] == df["1"]).sum()

        n_all +=len(df)
        if subtype == "R":
            correct += R_count      
        elif subtype == "F":
            correct += F_count
        else:
            raise ValueError("subtypeに予期せぬ値")

        print(R_count, F_count, eq_count)
    acc = correct/n_all
    print("acc:", acc)

scatter("05_23")
