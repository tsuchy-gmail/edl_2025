import os
import pandas as pd

patch_size = 896
stride = 896
csv_path = f"csv/size{patch_size}_stride{stride}/patch_info.csv"

df = pd.read_csv(csv_path)
df_by_subtype = df.groupby("subtype")
n_patchs_by_subtype = df_by_subtype["n_patchs"].sum()
print(n_patchs_by_subtype)
