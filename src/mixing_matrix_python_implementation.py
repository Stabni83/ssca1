import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


def mixing_matrix_estimation_by_hierachical_clustering(X, Nsources, Mmics, G=0):
    """
    Estimates mixing matrix using hierarchical clustering on sparse TF points.

    Args:
        X: Sparse time-frequency points (L, Mmics)
        Nsources: Number of sources
        Mmics: Number of microphones
        G: Clustering parameter

    Returns:
        Tuple of three mixing matrix estimates from different clustering stages
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, Mmics)

    # map points to one side
    Xrhs = X.copy()
    left_idx = X[:, 0] < 0
    Xrhs[left_idx, :] *= -1

    # compute transformed cosine distances as in original MATLAB
    d = pdist(X, metric="cosine")  # d = 1 - cos
    Y = 1.0 - np.abs(1.0 - d)
    Z = linkage(Y, method="average")

    lengthX = X.shape[0]
    Nc = Nsources - 1
    sorted_T_sum_array = np.zeros(int(max(Nsources, 2) + 10))
    sorted_T_sum_array[0] = 1.0

    # Adaptive clustering loop to find optimal number of clusters
    while True:
        if (
            sorted_T_sum_array[Nsources - 1]
            >= np.mean(sorted_T_sum_array[: Nsources - 1]) * 0.05
        ):
            break
        if Nc >= 10 * Nsources or lengthX < Nc:
            break
        if Nc > Nsources and np.sum(sorted_T_sum_array[Nsources:]) < 0.05 * np.sum(
            sorted_T_sum_array[:Nsources]
        ):
            break

        Nc += 1
        if Nc > 2 * Nsources:
            print("Warning: Nc>2*Nsources")

        T = fcluster(Z, t=Nc, criterion="maxclust")

        # Update cluster statistics
        counts = np.array([np.sum(T == t) for t in range(1, Nc + 1)])
        sorted_T_sum_array[: len(counts)] = np.sort(counts)[::-1]

    # Return NaN if no clusters found
    if "T" not in locals():
        H_nan = np.full((Mmics, Nsources), np.nan)
        return H_nan, H_nan.copy(), H_nan.copy()

    # First clustering stage: compute centroids
    counts = np.array([np.sum(T == t) for t in range(1, Nc + 1)])
    sorted_idx = np.argsort(-counts)
    centroid = np.zeros((Nsources, Mmics))
    for n in range(Nsources):
        cluster_label = sorted_idx[n] + 1
        mask = T == cluster_label
        if np.any(mask):
            centroid[n, :] = np.mean(Xrhs[mask, :], axis=0)
        else:
            centroid[n, :] = 0.0
    H_est_first_clustering = centroid.T

    # Outlier elimination and second clustering stage
    Xnew_list = []
    scale_std = 0.5
    centroid_eli = np.zeros((Nsources, Mmics))
    for n in range(Nsources):
        cluster_label = sorted_idx[n] + 1
        mask = T == cluster_label
        Xcluster_rhs = Xrhs[mask, :]
        if Xcluster_rhs.size == 0:
            centroid_eli[n, :] = 0.0
            continue

        # Remove outliers based on angular deviation
        numer = np.dot(Xcluster_rhs, centroid[n, :])
        denom = np.linalg.norm(Xcluster_rhs, axis=1) * (
            np.linalg.norm(centroid[n, :]) + 1e-12
        )
        one_minus_CosTheta = 1.0 - (numer / (denom + 1e-12))
        thr = (
            scale_std * np.std(one_minus_CosTheta)
            if Xcluster_rhs.shape[0] > 1
            else np.inf
        )
        points_taken_idx = np.where(one_minus_CosTheta < thr)[0]

        if points_taken_idx.size > 0:
            Xcluster_kept = Xcluster_rhs[points_taken_idx, :]
        else:
            Xcluster_kept = Xcluster_rhs.copy()

        centroid_eli[n, :] = np.mean(Xcluster_kept, axis=0)

        # Prepare cleaned data for second clustering
        Xcluster_orig = X[mask, :]
        if (
            points_taken_idx.size > 0
            and Xcluster_orig.shape[0] >= points_taken_idx.max() + 1
        ):
            Xcluster_selected = Xcluster_orig[points_taken_idx, :]
            Xnew_list.append(Xcluster_selected)

    H_est_after_elimination = centroid_eli.T

    if len(Xnew_list) == 0:
        H_est_second_clustering = np.full((Mmics, Nsources), np.nan)
        return H_est_first_clustering, H_est_after_elimination, H_est_second_clustering

    Xnew = np.vstack(Xnew_list)
    if Xnew.shape[0] < Nsources:
        H_est_second_clustering = np.full((Mmics, Nsources), np.nan)
        return H_est_first_clustering, H_est_after_elimination, H_est_second_clustering

    # Second clustering stage on cleaned data
    X2 = Xnew.copy()
    Xrhs2 = X2.copy()
    left_idx2 = X2[:, 0] < 0
    Xrhs2[left_idx2, :] *= -1

    d2 = pdist(X2, metric="cosine")
    Y2 = 1.0 - np.abs(1.0 - d2)
    Z2 = linkage(Y2, method="average")

    lengthX2 = X2.shape[0]
    Nc = Nsources - 1
    sorted_T_sum_array = np.zeros(max(Nsources, 2) + 10)
    sorted_T_sum_array[0] = 1.0

    while True:
        if (
            sorted_T_sum_array[Nsources - 1]
            >= np.mean(sorted_T_sum_array[: Nsources - 1]) * 0.05
        ):
            break
        if Nc >= 10 * Nsources or lengthX2 < Nc:
            break
        if Nc > Nsources and np.sum(sorted_T_sum_array[Nsources:]) < 0.05 * np.sum(
            sorted_T_sum_array[:Nsources]
        ):
            break

        Nc += 1
        if Nc > 2 * Nsources:
            print("Warning: Nc>2*Nsources (stage2)")

        T2 = fcluster(Z2, t=Nc, criterion="maxclust")
        counts2 = np.array([np.sum(T2 == t) for t in range(1, Nc + 1)])
        sorted_T_sum_array[: len(counts2)] = np.sort(counts2)[::-1]

    sorted_idx2 = np.argsort(-counts2)
    centroid_second = np.zeros((Nsources, Mmics))
    for n in range(Nsources):
        cluster_label = sorted_idx2[n] + 1
        mask2 = T2 == cluster_label
        if np.any(mask2):
            centroid_second[n, :] = np.mean(Xrhs2[mask2, :], axis=0)
        else:
            centroid_second[n, :] = 0.0

    H_est_second_clustering = centroid_second.T
    return H_est_first_clustering, H_est_after_elimination, H_est_second_clustering


def Mixing_matrix_estimation(
    X, K, signal_size, overlap, win, Nsources, Mmics, deltaTheta
):
    """
    Estimates mixing matrix from multi-channel audio using sparse TF analysis.

    Args:
        X: Multi-channel input signal (Mmics, signal_size)
        K: FFT window length
        signal_size: Total signal length
        overlap: Overlap ratio between frames
        win: Window function
        Nsources: Number of sources
        Mmics: Number of microphones
        deltaTheta: Angular threshold for sparse point selection

    Returns:
        Tuple of three mixing matrix estimates from different clustering stages
    """
    # Process only positive frequencies
    half_of_K = K // 2 + 1

    # Initialize buffers for STFT processing
    frame_real = np.zeros((Mmics, half_of_K))
    frame_imag = np.zeros((Mmics, half_of_K))
    real_fft_array = []
    imag_fft_array = []

    # Frame-based STFT processing
    b = 0
    step = int(K * overlap)
    for n in range(0, signal_size - K, step):
        b += 1
        for m in range(Mmics):
            xframe = X[m, n : n + K] * win
            temp = np.fft.fft(xframe, n=K)
            frame_real[m, :] = temp[:half_of_K].real
            frame_imag[m, :] = temp[:half_of_K].imag
        real_fft_array.append(frame_real.copy())
        imag_fft_array.append(frame_imag.copy())

    real_fft_array = np.array(real_fft_array)
    imag_fft_array = np.array(imag_fft_array)
    real_fft_array = np.transpose(real_fft_array, (1, 0, 2))
    imag_fft_array = np.transpose(imag_fft_array, (1, 0, 2))

    B = real_fft_array.shape[1]

    # Select most informative frequency bins by variance
    V = np.var(real_fft_array[0, :, :], axis=0)
    IDXk = np.argsort(V)[::-1]

    X_sparse_points = []

    # Collect sparse time-frequency points meeting angular criteria
    for k in range(min(80, half_of_K - 1)):
        Rk = real_fft_array[:, :, IDXk[k]]
        Ik = imag_fft_array[:, :, IDXk[k]]

        # Select points with small phase differences
        num = np.sum(Rk * Ik, axis=0)
        denom = np.sqrt(np.sum(Rk**2, axis=0)) * np.sqrt(np.sum(Ik**2, axis=0))
        one_minus_abs_cos = 1 - np.abs(num / (denom + 1e-12))

        sparse_points_idx = np.where(
            one_minus_abs_cos < (1 - np.cos(np.deg2rad(deltaTheta)))
        )[0]

        if sparse_points_idx.size > 0:
            from_R = Rk[:, sparse_points_idx]
            from_I = Ik[:, sparse_points_idx]

            # Filter out low-amplitude points
            mag_R = np.sqrt(np.sum(from_R**2, axis=0))
            high_idx_R = np.where(mag_R > 0.25)[0]
            from_R = from_R[:, high_idx_R]

            mag_I = np.sqrt(np.sum(from_I**2, axis=0))
            high_idx_I = np.where(mag_I > 0.25)[0]
            from_I = from_I[:, high_idx_I]

            X_sparse_points.append(from_R)

    if len(X_sparse_points) == 0:
        return (
            np.nan * np.ones((Mmics, Nsources)),
            np.nan * np.ones((Mmics, Nsources)),
            np.nan * np.ones((Mmics, Nsources)),
        )

    # Prepare sparse points for clustering
    X_sparse_points = np.concatenate(X_sparse_points, axis=1).T

    # Perform hierarchical clustering on sparse points
    H_est_first_clustering, H_est_after_elimination, H_est_second_clustering = (
        mixing_matrix_estimation_by_hierachical_clustering(
            X_sparse_points, Nsources, Mmics, G=0
        )
    )

    return H_est_first_clustering, H_est_after_elimination, H_est_second_clustering
