import numpy as np
from weight import weights_estimation
from proximal_operators import prox_weighted_l1, prox_constraint

def douglas_rachford_linear_scale(x_stft, A_est, gamma=0.1, tol=1e-5, max_iter=1000):
    M, F, T = x_stft.shape
    N = A_est.shape[1]
    x_flat = x_stft.reshape(M, -1)
    s = np.zeros((N, F * T), dtype=np.complex128)
    z = np.zeros_like(s)
    W = weights_estimation(x_flat, A_est)

    for i in range(max_iter):
        s_old = s.copy()
        s = prox_constraint(z, A_est, x_flat)
        y = prox_weighted_l1(2 * s - z, gamma, W)
        z = z + y - s

        rel_error = np.linalg.norm(s - s_old, 'fro') / (np.linalg.norm(s, 'fro') + 1e-8)
        if rel_error < tol:
            print(f"Douglas-Rachford converged after {i+1} iterations (Linear Scale).")
            break
    else:
        print(f"Douglas-Rachford reached max iterations ({max_iter}) (Linear Scale).")

    s_estimated = s.reshape(N, F, T)
    print(f"Sources estimated (Linear Scale): shape -> {s_estimated.shape}")
    return s_estimated, W

def douglas_rachford_mel_scale(x_mel, A_est, gamma=0.1, tol=1e-5, max_iter=1000):
    M, F_mel, T = x_mel.shape
    N = A_est.shape[1]
    x_flat_mel = x_mel.reshape(M, -1)
    s_mel = np.zeros((N, F_mel * T))
    z_mel = np.zeros_like(s_mel)
    W_mel = weights_estimation(x_flat_mel, A_est)

    for i in range(max_iter):
        s_old_mel = s_mel.copy()
        s_mel = prox_constraint(z_mel, A_est, x_flat_mel)
        y_mel = prox_weighted_l1(2 * s_mel - z_mel, gamma, W_mel)
        z_mel = z_mel + y_mel - s_mel

        rel_error = np.linalg.norm(s_mel - s_old_mel, 'fro') / (np.linalg.norm(s_mel, 'fro') + 1e-8)
        if rel_error < tol:
            print(f"Douglas-Rachford converged after {i+1} iterations (Mel Scale).")
            break
    else:
        print(f"Douglas-Rachford reached max iterations ({max_iter}) (Mel Scale).")

    s_estimated_mel = s_mel.reshape(N, F_mel, T)
    print(f"Sources estimated (Mel Scale): shape -> {s_estimated_mel.shape}")
    return s_estimated_mel, W_mel