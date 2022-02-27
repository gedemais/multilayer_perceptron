import numpy as np
import math
from activations import softmax


def mean_squared_error(p, y):
    return math.pow((p - y), 2)


def mean_squared_error_derivative(a, y):
    return 2.0 * (a - y)


def cross_entropy_error(p, y):
    return (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def cross_entropy_error_derivative(a, y):
    epsilon = 1e-8
    return -y / (a + epsilon) + (1.0 - y) / (1.0 - a + epsilon)
