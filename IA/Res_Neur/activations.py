import numpy as np

# activation function and its derivative
def tanh(x):
  return np.tanh(x)

def tanh_prime(x):
  return 1-np.tanh(x)**2

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig
