import random
import os
import torch

a = [1,3,4]

print(os.cpu_count())
print(torch.cuda.get_device_name(0))
