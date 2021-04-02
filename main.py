# Original author: Morgan McKinney 3/2021

import read
import neural_network
import numpy as np

# Initialize variables
train_data = []
test_data = []
class_index = 1
correct_predictions = 0
hidden_layer_count = 0
hidden_node_count = 0

# User input, read and store input csv files
print("MULTILAYER PERCEPTRON \n")
train_file = input("Enter training csv file name: ")
test_file = input("Enter testing csv file name: ")
train_data = read.read_file(train_file, train_data)
test_data = read.read_file(test_file, test_data)

# Store data row-wise
train_data = np.array(train_data)
test_data = np.array(test_data)

# Train
# Backpropagation

# Test
# Predict labels

# Report overall accuracy, percentage of test set accurately predicted
accuracy = (correct_predictions / len(test_data)) * 100
print("Overall accuracy: " + str(accuracy))
# Report architecture
print("Number of hidden layers: " + str(hidden_layer_count))
print("Number of hidden nodes: " + str(hidden_node_count))
print("Activation function used: Sigmoid")
