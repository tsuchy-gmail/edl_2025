import numpy as np

vecs = np.array([[2,1], [3,1], [5,1], [10,0]])
cov_matrix = np.cov(vecs.T, bias=False)
print(cov_matrix)
