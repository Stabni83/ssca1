import numpy as np


def wiener_mask(s, p=2.0):
    """
    Computes Wiener masks for source separation.

    Args:
        s: Source estimates (N, F, T)
        p: Power parameter for magnitude weighting

    Returns:
        Time-frequency masks (N, F, T)
    """
    mag_sources = np.abs(s) ** p
    denom = np.sum(mag_sources, axis=0, keepdims=True) + 1e-8
    masks = mag_sources / denom
    return masks
