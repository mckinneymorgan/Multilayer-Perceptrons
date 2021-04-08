# Original author: Morgan McKinney 3/2021
# Modified to handle MNIST dataset

import csv


# Read a given csv file
def read_file(file, data):
    # Read file
    with open(file) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        data = list(read_csv)

    return data


# Normalize feature values to be 0 or 1
def normalize(values):
    new_values = [int(x)/255 for x in values]
    for i, item in enumerate(new_values):
        if new_values[i] >= 0.5:
            new_values[i] = 1
        else:
            new_values[i] = 0
    return new_values
