# Original author: Morgan McKinney 3/2021

import read
import neural_network
import numpy as np
from random import random

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
output_node_count = 1
total_layer_count = hidden_layer_count + 2  # Add input and output layers

# User input, read and store input csv files
print("MULTILAYER PERCEPTRON \n")
train_file = input("Enter training csv file name: ")
test_file = input("Enter testing csv file name: ")
train_data = read.read_file(train_file, train_data)
test_data = read.read_file(test_file, test_data)

# Separate class labels from features
train_features = train_data[1:]
train_labels = [element[0] for element in train_data]
test_features = test_data[1:]
test_labels = [element[0] for element in test_data]

# Normalize feature values
train_features = [read.normalize(element) for element in train_features]
test_features = [read.normalize(element) for element in test_features]

# Store features row-wise, convert lists to arrays
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# Initialize weights and biases
weights = []
biases = []
for n in range(total_layer_count):
    # Input layer
    if n == 0:
        weights.append(np.full((len(train_features), hidden_node_count), random()))
        biases.append(np.full(hidden_node_count, random()))
    # Hidden layer(s)
    elif n != total_layer_count-1:
        weights.append(np.full((hidden_node_count, hidden_node_count), random()))
        biases.append(np.full(hidden_node_count, random()))
    # Output layer
    else:
        weights.append(np.full((hidden_node_count, output_node_count), random()))
        biases.append(np.full(output_node_count, random()))
print(weights)
print(biases)

# Train
for epoch in range(epochMax):
    meanSquaredError = []
    for n, example in enumerate(train_features):
        # Forward pass
        inputs = train_features[n]
        ground_truth = train_labels[n]
        # Compute output
        # Calculate deltas
        # Backpropagation
        # Calculate gradient
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
