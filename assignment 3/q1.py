### Solution Q1 C ####
import numpy as np
from sklearn.datasets import load_boston

def gradient(x, y, w):

    n = x.shape[0]
    xTxw = np.dot(np.dot(np.transpose(x), x), w)
    xTy = np.dot(np.transpose(x), y)
    gradient = np.divide((2 * xTxw - 2 * xTy), n)
    return gradient

# load boston housing prices dataset

x, y = load_boston(True)
d = x.shape[1]
w = np.zeros(d) 

grad= gradient(x, y, w)
print (grad)

