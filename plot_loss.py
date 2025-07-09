import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

#csv_path_vir = "result/2025/07_01/crop224/lamb0to1/0018_20715/test_loss.csv"
#csv_path_non = "result/2025/06_09/crop224/lamb0to1/0141_02155/test_loss.csv"
csv_path = "result/2025/07_07/crop224/lamb0to1/1726_06380/test_loss.csv"
df = pd.read_csv(csv_path)
plot_acc(df, "Accuracy", "result/2025/07_07/crop224/lamb0to1/1726_06380/acc.png")
