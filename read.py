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
