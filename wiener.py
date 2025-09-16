import numpy as np

def wiener_mask(s, p=2.0):
    mag_sources = np.abs(s) ** p
    denom = np.sum(mag_sources, axis=0, keepdims=True) + 1e-8
    masks = mag_sources / denom
    return masks