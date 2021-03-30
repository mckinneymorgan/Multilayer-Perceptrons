# Original author: Morgan McKinney 3/2021

import read
import sys
from random import random
import numpy as np
import matplotlib.pyplot as plt

# Initialize variables
data = []
names = []

# User input, read and store input csv file
print("MULTILAYER PERCEPTRON \n")
read.read_file(names, data)
data = np.array(data)  # Store data row-wise
