import numpy as np
from math import exp

def softmax(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

#def dsoftmax(x)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid(x):
  return 1 / (1 + exp(-x))

#def dsigmoid(x)
