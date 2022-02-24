import numpy as np
import math

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def dsoftmax(inputs):
    sft = np.empty(inputs.shape)
    for i in range(inputs.shape[0]):
        xsum = np.sum(np.exp(inputs[i]))
        sft[i, ...] = np.exp(inputs[i, ...]) / xsum
    return sft * (1. - sft)


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1.0 * (x > 0)


def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(x):
    return x * (1.0 - x)
