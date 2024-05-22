import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    e_x = np.exp(x-np.max(x,axis=-1,keepdims=True))
    return e_x/np.sum(e_x,axis=-1, keepdims=True)

def softmax_prime(x):
    s = softmax(x)
    # Non vectorized version
    # n = x.shape[-1]
    # dX = np.empty((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         if i==j:
    #             dX[i,j] = s[:,i]*(1-s[:,j])
    #         else:
    #             dX[i,j] = -s[:,i]*s[:,j]
    # Vectorized version
    s = s.reshape(-1, 1)
    dX = np.diagflat(s) - np.dot(s, s.T)
    return dX