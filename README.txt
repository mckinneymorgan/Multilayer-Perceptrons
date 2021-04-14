README for Multilayer Perceptrons

Original author: Morgan McKinney 3/2021

Machine learning binary classification using multilayer perceptron on the MNIST handwriting dataset. Outputs the accuracy of the model alongside network architecture.

Training set and testing set are inputted by the user at the start of the program. Expected dataset entry formatting is a flattened list of values ranging from 0 to 255. The class labels should be at the first index of each entry, indicating which number or letter the row corresponds to. Finally, the file is expected to be of type .csv.

This program is designed to be generalized such that it can handle interchangable amounts of hidden layers, hidden nodes, and output nodes. Furthermore, the learning rate and training time (epochs) can be tuned if desired. Due to the simplistic nature of the learning problem, the activation function is fixed as Sigmoid.

main.py: Contains hyperparameters alongside training and testing implementation.

neural_network.py: Activation function of neural network.

read.py: Reads any given .csv file and normalizes data for streamlined handling.
