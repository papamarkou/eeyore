import numpy as np

def softabs_np(hessian, a=1000.0):
    l, Q = np.linalg.eig(hessian)
    return np.matmul(np.matmul(Q, np.diag(l/(np.tanh(a*l)))), Q.transpose())
