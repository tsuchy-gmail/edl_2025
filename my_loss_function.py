import torch
import torch.nn.functional as F

def my_dirichlet_reg(alpha, y):
    alpha = y + (1 - y) * alpha
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl

def my_dirichlet_mse(alpha, y):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    return t1 + t2


def my_evidential_classification(alpha, y):
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    return my_dirichlet_mse(alpha, y), my_dirichlet_reg(alpha, y)

