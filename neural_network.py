# Original author: Morgan McKinney 4/2021

import numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
