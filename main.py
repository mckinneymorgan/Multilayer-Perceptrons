# Original author: Morgan McKinney 3/2021

import read
import neural_network
import numpy as np

# Initialize variables
train_data = []
test_data = []
class_index = 1
correct_predictions = 0

# Hyperparameters, tune as needed
alpha = 0.5
epochMax = 10
hidden_layer_count = 1
hidden_node_count = 2  # Same number of hidden nodes per layer

# User input, read and store input csv files
print("MULTILAYER PERCEPTRON \n")
train_file = input("Enter training csv file name: ")
test_file = input("Enter testing csv file name: ")
train_data = read.read_file(train_file, train_data)
test_data = read.read_file(test_file, test_data)

# Separate class labels from features
train_features = read.class_labels(train_data)
test_features = read.class_labels(test_data)

# Normalize data

# Store data row-wise
train_data = np.array(train_data)
test_data = np.array(test_data)

# Train
# Compute output
inputs = train_data[1:]  # Copy all data except class label

# Calculate deltas
# Backpropagation
# Update weights
# Weight updates

# Test
# Predict labels

# Report overall accuracy, percentage of test set accurately predicted
accuracy = (correct_predictions / len(test_data)) * 100
print("Overall accuracy: " + str(accuracy))
# Report architecture
print("Number of hidden layers: " + str(hidden_layer_count))
print("Number of hidden nodes: " + str(hidden_node_count))
print("Activation function used: Sigmoid")
