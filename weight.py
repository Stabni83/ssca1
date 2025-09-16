import numpy as np

def weights_estimation(x , a , eps = 1e-8):
    M , F , Q = x.shape
    N = a.shape[1]
    x_flat = x.reshape(M , -1)
    w = np.zeros((N , F * Q))
    for i in range(F * Q):
        x = x_flat[: , i]
        x_norm = np.linalg.norm(x) + eps
        for n in range(N):
            a_n = a[: , n]
            a_n_norm = np.linalg.norm(a_n) + eps
            temp = (np.abs(np.dot(x.T , a_n))) / (x_norm * a_n_norm)
            temp = np.clip(temp , 0 , 1)
            w[n , i] = a_n_norm * np.sqrt(1 - temp ** 2)
    print(f"weights computed : shape -> {w.shape}")
    return w
