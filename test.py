import numpy as np
import os

a = [[2,1], [3,4], [1,1], [10,1]]
a_np = np.array(a)
print(a_np)
print(a_np.shape)
print(type(a_np))

b = np.load("test.npz")
print(b["a"])

points = [(1, 2), (3, 4), (5, 6)]
x, y = zip(*points)
print(x)
print(y)
