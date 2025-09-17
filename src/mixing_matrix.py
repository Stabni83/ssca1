import numpy as np
from mixing_matrix_python_implementation import Mixing_matrix_estimation

Mmics = 2
G = 0
signal_size = 16000 * 10 
K = 1024
overlap = 128 / 1024
deltaTheta = 0.05
B = int(np.floor((signal_size + overlap * K - K) / (overlap * K)))
win = np.hanning(K).tolist()  
H_empty_3_source = [[1,1,1],[1,1,1]]
H_empty_4_source = [[1,1,1,1],[1,1,1,1]]

def A_estimation(x , n_sources):
    A_est = outputs = Mixing_matrix_estimation(
        x,
        K,            
        signal_size,overlap,     
        win,            
        n_sources,Mmics,       
        deltaTheta  
    )[-1]
    A_est = A_est / (np.linalg.norm(A_est, axis=0, keepdims=True) + 1e-8)
    print(A_est)
    return A_est