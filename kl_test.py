import random
import os
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
a = [1, 3]
b = [2, 1]
if True:
    cc = 1
print(cc)
exit(0)

# 行ごとに並べるには zip(*[a, b]) を使って「転置」する

alpha = np.array([[2,1],[1,3],[10,1]])

alpha_R = alpha[alpha[:, 0] > alpha[:, 1]]

unc = 2 / (alpha[:, 0] + alpha[:, 1])

def my_evidential_classification(alpha, y):
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    return dirichlet_reg(alpha, y)

def dirichlet_mse(alpha, y):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    return t1 + t2

def dirichlet_reg(alpha, y):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl

label = torch.tensor([1, 1, 0, 0, 1, 1, 0, 1])
y = torch.from_numpy(np.array([[0,1], [0,1], [1,0], [1,0], [0,1], [0,1], [1,0], [0,1]]))
alpha = torch.from_numpy(np.array([[1.010,1.001], [1, 3], [1, 3], [100, 1], [100, 1],[26,1],[26,1], [10, 10]]))

kl = dirichlet_reg(alpha, y)
kl_my = my_evidential_classification(alpha, label)

print(kl_my)
