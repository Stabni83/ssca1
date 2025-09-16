import numpy as np
from mixing_matrix_python_implementation import Mixing_matrix_estimation

def A_estimation(x, n_sources):
    Mmics = 2
    signal_size = x.shape[1]
    K = 1024
    overlap = 128 / 1024
    win = np.hanning(K).tolist()
    deltaTheta = 0.05

    A_est = Mixing_matrix_estimation(
        x,
        K,
        signal_size, overlap,
        win,
        n_sources, Mmics,
        deltaTheta
    )[-1]
    A_est = A_est / (np.linalg.norm(A_est, axis=0, keepdims=True) + 1e-8)
    print("Estimated Mixing Matrix A:\n", A_est)
    return A_est