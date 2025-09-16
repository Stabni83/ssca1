import numpy as np
from weight import weights_estimation
from proximal_operators import f1 , f2

def douglas_rachford(x , a , gamma = 0.1 , tol = 1e-5 , max_iteration = 10000):
    M , F , Q = x.shape
    N = a.shape[1]
    x_flat = x.reshape(M , -1)
    s = np.zeros((N , F * Q))
    z = np.zeros_like(s)
    w = weights_estimation(x , a)
    for i in range(max_iteration):
        s_old = s.copy()
        s = f2(z , a , x_flat)
        y = f1(2 * s - z , gamma , w)
        z = z + y - s
        if (np.linalg.norm((s - s_old) , "fro") / np.linalg.norm(s , "fro")) < tol:
            print(f"Converged after {i+1} iterations")
            break
    print(f"sources has been estimated : shape -> {s.reshape(N, F, Q).shape}")
    return s.reshape(N, F, Q)