import numpy as np
from weight import weights_estimation
from proximal_operators import prox_weighted_l1, prox_constraint


def douglas_rachford_linear_scale(x_stft, A_est, gamma=0.1, tol=1e-5, max_iter=1000):
    """
    Douglas-Rachford splitting algorithm for source separation in linear frequency scale.
    Solves the weighted L1 minimization problem with linear constraints.

    Args:
        x_stft (numpy.ndarray): STFT of mixture signal, shape (M, F, T)
        A_est (numpy.ndarray): Estimated mixing matrix, shape (M, N)
        gamma (float): Step size parameter for proximal operators
        tol (float): Tolerance for convergence checking
        max_iter (int): Maximum number of iterations

    Returns:
        tuple: (s_estimated, W) where:
            - s_estimated: Estimated sources in time-frequency domain, shape (N, F, T)
            - W: Weight matrix used for weighted L1 regularization
    """
    M, F, T = x_stft.shape
    N = A_est.shape[1]

    # Flatten the STFT matrix for processing
    x_flat = x_stft.reshape(M, -1)

    # Initialize source estimates and auxiliary variable
    s = np.zeros((N, F * T), dtype=np.complex128)
    z = np.zeros_like(s)

    # Estimate weights based on signal geometry
    W = weights_estimation(x_flat, A_est)

    # Main optimization loop
    for i in range(max_iter):
        s_old = s.copy()

        # Douglas-Rachford update steps
        s = prox_constraint(z, A_est, x_flat)  # Project onto constraint set
        y = prox_weighted_l1(2 * s - z, gamma, W)  # Apply weighted L1 proximal
        z = z + y - s  # Update auxiliary variable

        # Check convergence
        rel_error = np.linalg.norm(s - s_old, "fro") / (np.linalg.norm(s, "fro") + 1e-8)
        if rel_error < tol:
            print(f"Douglas-Rachford converged after {i+1} iterations (Linear Scale).")
            break
    else:
        print(f"Douglas-Rachford reached max iterations ({max_iter}) (Linear Scale).")

    # Reshape estimated sources to original dimensions
    s_estimated = s.reshape(N, F, T)
    print(f"Sources estimated (Linear Scale): shape -> {s_estimated.shape}")

    return s_estimated, W


def douglas_rachford_mel_scale(x_mel, A_est, gamma=0.1, tol=1e-5, max_iter=1000):
    """
    Douglas-Rachford splitting algorithm for source separation in Mel frequency scale.
    Optimized version for Mel-scale spectrograms with reduced frequency resolution.

    Args:
        x_mel (numpy.ndarray): Mel-spectrogram of mixture signal, shape (M, F_mel, T)
        A_est (numpy.ndarray): Estimated mixing matrix, shape (M, N)
        gamma (float): Step size parameter for proximal operators
        tol (float): Tolerance for convergence checking
        max_iter (int): Maximum number of iterations

    Returns:
        tuple: (s_estimated_mel, W_mel) where:
            - s_estimated_mel: Estimated sources in Mel-scale, shape (N, F_mel, T)
            - W_mel: Weight matrix used for weighted L1 regularization in Mel-scale
    """
    M, F_mel, T = x_mel.shape
    N = A_est.shape[1]

    # Flatten the Mel-spectrogram for processing
    x_flat_mel = x_mel.reshape(M, -1)

    # Initialize source estimates and auxiliary variable
    s_mel = np.zeros((N, F_mel * T))
    z_mel = np.zeros_like(s_mel)

    # Estimate weights based on Mel-scale signal geometry
    W_mel = weights_estimation(x_flat_mel, A_est)

    # Main optimization loop
    for i in range(max_iter):
        s_old_mel = s_mel.copy()

        # Douglas-Rachford update steps
        s_mel = prox_constraint(z_mel, A_est, x_flat_mel)  # Project onto constraint set
        y_mel = prox_weighted_l1(
            2 * s_mel - z_mel, gamma, W_mel
        )  # Apply weighted L1 proximal
        z_mel = z_mel + y_mel - s_mel  # Update auxiliary variable

        # Check convergence
        rel_error = np.linalg.norm(s_mel - s_old_mel, "fro") / (
            np.linalg.norm(s_mel, "fro") + 1e-8
        )
        if rel_error < tol:
            print(f"Douglas-Rachford converged after {i+1} iterations (Mel Scale).")
            break
    else:
        print(f"Douglas-Rachford reached max iterations ({max_iter}) (Mel Scale).")

    # Reshape estimated sources to original dimensions
    s_estimated_mel = s_mel.reshape(N, F_mel, T)
    print(f"Sources estimated (Mel Scale): shape -> {s_estimated_mel.shape}")

    return s_estimated_mel, W_mel
