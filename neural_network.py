# Original author: Morgan McKinney 4/2021

import math


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))
