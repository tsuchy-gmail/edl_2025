import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("csv/patch_count.csv")
bins = np.logspace(np.log10(df["count"].min()), np.log10(df["count"].max()), num=10)

plt.hist(df["count"], bins=bins, edgecolor="black")
plt.xlabel("Number of Patch")
plt.ylabel("Number of Cases")
plt.xscale("log")
plt.tight_layout()
plt.savefig("figure/histogram_log.png")


