import numpy as np


def epsilon_equal(a, b, epsilon=1e-4):
    return np.sum(np.abs(a - b)) < epsilon
