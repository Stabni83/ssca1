import numpy as np


def weights_estimation(x_flat, a, eps=1e-8):
    """
    Estimates weights based on angular similarity between mixture and source directions.

    Args:
        x_flat: Flattened mixture signal (M, T)
        a: Mixing matrix (M, N)
        eps: Small value for numerical stability

    Returns:
        Weight matrix (N, T) based on sine of angles
    """
    M, T = x_flat.shape
    N = a.shape[1]
    w = np.zeros((N, T))

    # Compute norms for normalization
    x_norms = np.linalg.norm(x_flat, axis=0) + eps
    a_norms = np.linalg.norm(a, axis=0) + eps

    # Calculate weights for each source
    for n in range(N):
        a_n = a[:, n]
        dot_products = np.abs(np.dot(a_n.T, x_flat))
        cos_theta = dot_products / (a_norms[n] * x_norms)
        cos_theta = np.clip(cos_theta, 0, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        w[n, :] = a_norms[n] * sin_theta

    print(f"Weights computed: shape -> {w.shape}")
    return w
