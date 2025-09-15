import numpy as np

def multi_scale_weights_estimation(x_multi_scale, a, eps=1e-8):
    """
    Multi-scale weight estimation with geometric mean combination
    """
    weight_list = []
    
    for x_stft in x_multi_scale:
        M, F, Q = x_stft.shape
        x_flat = x_stft.reshape(M, -1)
        N = a.shape[1]
        T = x_flat.shape[1]
        w_scale = np.zeros((N, T))
        
        for i in range(T):
            x_vec = x_flat[:, i]
            x_norm = np.linalg.norm(x_vec) + eps
            for n in range(N):
                a_n = a[:, n]
                a_n_norm = np.linalg.norm(a_n) + eps
                cos_theta = np.abs(np.dot(x_vec.T, a_n)) / (x_norm * a_n_norm)
                cos_theta = np.clip(cos_theta, 0, 1)
                w_scale[n, i] = a_n_norm * np.sqrt(1 - cos_theta**2)
        
        weight_list.append(w_scale)
    
    
    stacked = np.stack(weight_list, axis=0)
    return np.exp(np.mean(np.log(stacked + 1e-8), axis=0))