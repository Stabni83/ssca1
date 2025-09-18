import numpy as np


def prox_weighted_l1(Z, gamma, W):
    """
    Proximal operator for the weighted L1 norm penalty term.
    Applies soft-thresholding with element-wise weights.

    Args:
        Z (numpy.ndarray): Input matrix
        gamma (float): Step size parameter
        W (numpy.ndarray): Weight matrix with same shape as Z

    Returns:
        numpy.ndarray: Thresholded matrix after applying weighted L1 proximal operator
    """
    return np.sign(Z) * np.maximum(np.abs(Z) - gamma * W, 0)


def prox_constraint(Z, A, X):
    """
    Proximal operator for the linear constraint X = A*S.
    Projects the input onto the feasible set {S: A*S = X}.

    Args:
        Z (numpy.ndarray): Current estimate of sources
        A (numpy.ndarray): Mixing matrix
        X (numpy.ndarray): Observed mixture signal

    Returns:
        numpy.ndarray: Projected source estimate that satisfies the constraint A*S = X
    """
    return Z + A.T @ np.linalg.inv(A @ A.T) @ (X - A @ Z)
