# Original author: Morgan McKinney 3/2021

import csv
import sys


# Read a given csv file and store in input variables
def read_file(names, data):
    class_value_file_exist = False
    feature_names = False
    class_value_name = "class"
    class_values = []

    # User input, get filename(s)
    file = input("Enter csv file name: ")
    class_value_file = input("Are the class values in another file (Y/N): ")
    if class_value_file.lower() == 'y':
        class_value_file = input("Enter class value file name: ")
        class_value_file_exist = True
    elif class_value_file.lower() != 'n':
        sys.exit("Invalid input")
    feature_names_exist = input("Are the features named (Y/N): ")
    if feature_names_exist.lower() == 'y':
        feature_names = True
    elif feature_names_exist.lower() != 'n':
        sys.exit("Invalid input")

    # Read file
    with open(file) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')

        # Count features, return to start of file
        feature_num = len(next(read_csv))
        if not class_value_file_exist:
            feature_num -= 1
        csvfile.seek(0)

        # Name features, if provided
        if feature_names:
            names = next(read_csv)
            if not class_value_file_exist:
                class_value_name = names[feature_num]
                names.pop()
        else:
            for x in range(feature_num):
                names.append(x)

        # Populate data list
        for row in read_csv:
            row_list = []
            for x in range(feature_num):
                row_entry = float(row[x])
                row_list.append(row_entry)
            data.append(row_list)

        # Populate class list
        if not class_value_file_exist:
            csvfile.seek(0)
            if feature_names:
                next(read_csv)
            for row in read_csv:
                class_values.append(row[feature_num])
    # Open class value file if applicable
    if class_value_file_exist:
        with open(class_value_file) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',')

            # Populate class list
            if feature_names_exist:
                class_value_name = next(read_csv)
            for row in read_csv:
                class_values.append(row[0])

    # Append class values to data set
    entry = -1
    for x in data:
        entry += 1
        class_label = class_values[entry]
        if class_label == 'True':
            class_label = 1
        elif class_label == 'False':
            class_label = 0
        else:
            if entry != 0 and not feature_names:
                class_label = float(class_label)
        data[entry].append(class_label)

    # Append class value name to name list
    names.append(class_value_name)
