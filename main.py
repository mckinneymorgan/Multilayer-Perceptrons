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
alpha = 0.0000000001
epochMax = 1
hidden_layer_count = 1
hidden_node_count = 2  # Same number of hidden nodes per layer
output_node_count = 1
total_layer_count = hidden_layer_count + 1  # Add output layer

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

# Store features row-wise, convert data to arrays
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
        size_weights = (len(test_features[n]), hidden_node_count)
        size_biases = (hidden_node_count, 1)
        weights.append(np.random.uniform(-1, 1, size_weights))
        biases.append(np.random.uniform(-1, 1, size_biases))
    # Hidden layer(s)
    elif n != hidden_layer_count:
        size_weights = (hidden_node_count, hidden_node_count)
        size_biases = (hidden_node_count, 1)
        weights.append(np.random.uniform(-1, 1, size_weights))
        biases.append(np.random.uniform(-1, 1, size_biases))
    # Output layer
    else:
        size_weights = (hidden_node_count, output_node_count)
        size_biases = (output_node_count, 1)
        weights.append(np.random.uniform(-1, 1, size_weights))
        biases.append(np.random.uniform(-1, 1, size_biases))
    print("Biases [" + str(n) + "]: " + str(biases[n].shape))
    print("Weights [" + str(n) + "]: " + str(weights[n].shape))

# Train
for epoch in range(epochMax):
    for i, example in enumerate(train_features):
        # Forward pass
        features = train_features[i]
        # print("X: " + str(features.shape))
        ground_truth = float(train_labels[i])
        outputs = []
        # Compute output
        for n in range(total_layer_count):
            if n == 0:  # Size hidden_node_count x 1
                product = np.transpose(weights[n]).dot(features)
            else:
                product = np.transpose(weights[n]).dot(outputs[n - 1])
            inputs = product + np.transpose(biases[n])
            inputs = np.transpose(inputs)
            neural_network.sigmoid(np.array(inputs))  # Activation
            outputs.append(inputs)
            # print("Outputs [" + str(n) + "]: " + str(outputs[n].shape))
        output = float(outputs[-1])  # Output of network
        # Calculate error
        error = ground_truth - output
        if i % 100 == 0:  # Print error in intervals
            print("Error: " + str(error))
        if -0.05 < error < 0.05:  # Stop training once low error achieved
            break
        # Backpropagation
        deltas = []
        # Calculate deltas
        for n in reversed(range(total_layer_count)):
            if n == hidden_layer_count:  # Scalar
                delta = error * (output * (1 - output))
            else:  # Size hidden_node_count x 1
                delta = weights[n + 1].dot(deltas[0]) * (outputs[n] * (1 - outputs[n]))
            deltas.insert(0, np.array(delta))
            # print("Delta [" + str(n) + "]: " + str(deltas[0].shape))
        # Update weights
        weights_temp = weights.copy()
        biases_temp = biases.copy()
        for n in reversed(range(total_layer_count)):
            if n == hidden_layer_count:
                weights[n] = weights_temp[n] + alpha * outputs[n - 1] * deltas[n]
                biases[n] = biases_temp[n] + alpha * deltas[n]
            else:
                weights[n] = weights_temp[n] + np.transpose(alpha * (features * deltas[n]))
                biases[n] = biases_temp[n] + alpha * deltas[n]
        # print("Weights:")
        # print(weights)
        # print("Biases:")
        # print(biases)

# Test
for i, example in enumerate(test_features):
    features = test_features[i]
    # print("X: " + str(features.shape))
    ground_truth = float(test_labels[i])
    outputs = []
    # Computer output
    for n in range(total_layer_count):
        if n == 0:  # Size hidden_node_count x 1
            product = np.transpose(weights[n]).dot(features)
        else:
            product = np.transpose(weights[n]).dot(outputs[n - 1])
        inputs = product + np.transpose(biases[n])
        inputs = np.transpose(inputs)
        neural_network.sigmoid(np.array(inputs))  # Activation
        outputs.append(inputs)
        # print("Outputs [" + str(n) + "]: " + str(outputs[n].shape))
    output = float(outputs[-1])  # Output of network
    prediction = round(output)
    if prediction >= 0.5:  # Normalize output
        prediction = 1
    else:
        prediction = 0
    # Check for accuracy
    if ground_truth == prediction:
        correct_predictions += 1

# Report overall accuracy, percentage of test set accurately predicted
accuracy = (float(correct_predictions) / float(len(test_data))) * 100
print("Overall accuracy: " + str(accuracy))
# Report architecture
print("Number of hidden layers: " + str(hidden_layer_count))
print("Number of hidden nodes: " + str(hidden_node_count))
print("Activation function used: Sigmoid")
