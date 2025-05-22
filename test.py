import random
import os
import torch

a = [1,3,4]

print("cpu_count:", os.cpu_count())
b, c = 2, 1

l = []

def ap(lst):
    lst.append(1)

print(l)
ap(l)
print(l)

with open("rmtar.txt", "w") as f:
    f.write("hello")
