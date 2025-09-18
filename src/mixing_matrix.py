import numpy as np
from mixing_matrix_python_implementation import Mixing_matrix_estimation

# Global parameters for mixing matrix estimation
Mmics = 2  # Number of microphones
G = 0  # Parameter for hierarchical clustering (0 for default)
signal_size = 16000 * 10  # Signal size for 10 seconds at 16kHz sampling rate
K = 1024  # FFT window size
overlap = 128 / 1024  # Overlap ratio between frames
deltaTheta = 0.05  # Threshold angle in degrees for sparse point selection
B = int(np.floor((signal_size + overlap * K - K) / (overlap * K)))  # Number of frames
win = np.hanning(K).tolist()  # Hanning window function

# Empty mixing matrix templates for initialization
H_empty_3_source = [[1, 1, 1], [1, 1, 1]]  # Template for 3 sources
H_empty_4_source = [[1, 1, 1, 1], [1, 1, 1, 1]]  # Template for 4 sources


def A_estimation(x, n_sources):
    """
    Estimates the mixing matrix A using hierarchical clustering on sparse time-frequency points.

    Args:
        x (numpy.ndarray): Input mixture signal of shape (channels, samples)
        n_sources (int): Number of sources to estimate

    Returns:
        numpy.ndarray: Estimated mixing matrix A of shape (Mmics, n_sources)
    """
    # Estimate mixing matrix using hierarchical clustering method
    A_est = outputs = Mixing_matrix_estimation(
        x, K, signal_size, overlap, win, n_sources, Mmics, deltaTheta
    )[
        -1
    ]  # Use the result from the second clustering stage

    # Normalize the estimated mixing matrix columns to unit norm
    A_est = A_est / (np.linalg.norm(A_est, axis=0, keepdims=True) + 1e-8)

    print("Estimated mixing matrix A:\n", A_est)
    return A_est
