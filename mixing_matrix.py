import matlab
import numpy as np 

Mmics = float(2) 
G = float(0)
signal_size = float(16000 * 10) 
K = float(1024)
overlap = float(128 / 1024) 
deltaTheta = float(0.05) 
B = float(int(np.floor((signal_size + overlap * K - K) / (overlap * K)))) 
win = np.hanning(K).tolist()  
H_empty_3_source = matlab.double([[1,1,1],[1,1,1]]) 
H_empty_4_source = matlab.double([[1,1,1,1],[1,1,1,1]]) 

def A_estimation(x , n_sources , eng):
    X_matlab = matlab.double(x.tolist())
    win_matlab = matlab.double([win]) 
    A_est = outputs = eng.Mixing_matrix_estimation(
        H_empty_3_source if n_sources == 3 else H_empty_4_source ,X_matlab,
        K,B,            
        signal_size,overlap,     
        win_matlab,G,            
        float(n_sources),Mmics,       
        deltaTheta,nargout=6  
    )[-1]
    A_est = A_est / (np.linalg.norm(A_est, axis=0, keepdims=True) + 1e-8)
    print(A_est)
    return A_est
