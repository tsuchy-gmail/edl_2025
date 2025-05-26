import random
import os
import torch
import pandas as pd


df = pd.read_csv("result/05_21/2025_0521_030342/train_loss.csv")
print(len(df))

df = pd.DataFrame([[1,2], [1,1.1], [2,1]], columns=["a", "b"])
print(df["a"])
normalized = df.div(df.sum(axis=1), axis=0)
print(normalized)

