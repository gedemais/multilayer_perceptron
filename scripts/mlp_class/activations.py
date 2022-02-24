import numpy as np
import math

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

#def dsoftmax(x)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#def dsigmoid(x)
