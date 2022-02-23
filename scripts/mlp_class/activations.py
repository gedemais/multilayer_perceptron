import numpy as np
from math import exp

def softmax(numbers):
    exponentials = np.exp(numbers)
    sum_exponentials = sum(exponentials)
    return exponentials/sum_exponentials

#def dsoftmax(x)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid(x):
  return 1 / (1 + exp(-x))

#def dsigmoid(x)
