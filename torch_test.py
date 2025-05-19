import torch

a = [1,0,1]
b = [2,4,7]

a_t = torch.tensor(a)
b_t = torch.tensor(b)

print(type(a_t == 1))
print(a_t == 1)
print(b_t[a_t])

