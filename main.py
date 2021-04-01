# Original author: Morgan McKinney 3/2021

import read
import numpy as np
# import matplotlib.pyplot as plt

# Initialize variables
train_data = []
test_data = []

# User input, read and store input csv files
print("MULTILAYER PERCEPTRON \n")
train_file = input("Enter training csv file name: ")
test_file = input("Enter testing csv file name: ")
train_data = read.read_file(train_file, train_data)
test_data = read.read_file(test_file, test_data)

# Store data row-wise
train_data = np.array(train_data)
test_data = np.array(test_data)
