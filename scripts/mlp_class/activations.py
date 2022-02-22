import numpy as np

def softmax(x):
    """Compute softmax value for x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
