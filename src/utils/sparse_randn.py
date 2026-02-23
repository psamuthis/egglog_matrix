import numpy as np

def sparse_randn(shape, sparsity):
    arr = np.random.randn(*shape)
    mask = np.random.random(shape) < sparsity
    arr[mask] = 0
    return arr