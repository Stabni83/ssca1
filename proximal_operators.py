import numpy as np

def f1(Z, gamma, W):
    return np.sign(Z) * np.maximum(np.abs(Z) - gamma * W, 0)

def f2(z , a , x):
    return z + a.T @ np.linalg.inv(a @ a.T) @ (x - a @ z)

