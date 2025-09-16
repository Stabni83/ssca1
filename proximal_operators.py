import numpy as np

def prox_weighted_l1(Z, gamma, W):
    return np.sign(Z) * np.maximum(np.abs(Z) - gamma * W, 0)

def prox_constraint(Z, A, X):
    return Z + A.T @ np.linalg.inv(A @ A.T) @ (X - A @ Z)