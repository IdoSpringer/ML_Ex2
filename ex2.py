import sys
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split

# todo:
# Tal if you have any question ask me on WhatsApp or Skype
# Please implement at least one training algorithm function
# (I think perceptron is the easiest)
# You can change the other code if needed


def load_data(data_file):
    data = []
    # Sex feature to numerical index dictionary
    sex_to_index = {'M': 0, 'F': 1, 'I': 2}
    # Read file
    with open(data_file, 'r') as file:
        for line in file:
            # Split entry
            line = line.strip().split(',')
            line[0] = sex_to_index[line[0]]
            # Add to data
            data.append(np.array(line, dtype=np.float64))
    return np.array(data)


def load_labels(labels_file):
    # Read labels from file
    labels = np.loadtxt(labels_file, dtype=np.float64)
    return labels


def normalize_data():
    # Optional, we will try normalization later
    pass


def train_dev_split(train_data, train_labels):
    # We split the original train file to 80% train and 20% validation data
    # We will delete this after hyper-parameters tuning and use only test file
    train_x, dev_x, train_y, dev_y = train_test_split(train_data, train_labels, test_size=0.2)
    return train_x, dev_x, train_y, dev_y


def train_perceptron(train_x, train_y, dev_x, dev_y):
    # Perceptron algorithm
    pass


def train_svm(train_x, train_y, dev_x, dev_y):
    # Support Vector Machine algorithm
    pass


def train_pa(train_x, train_y, dev_x, dev_y):
    # Passive Aggressive algorithm
    pass


def evaluate(dev_x, dev_y, W):
    # compute dev accuracy using trained parameters
    pass


def predict(test_x, trained_parameters):
    pass


def main(argv):
    train_data = load_data(argv[1])
    train_labels = load_labels(argv[2])
    train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
    # Train parameters with 3 algorithms
    # Predict test and print prediction
    pass


# main
if __name__ == '__main__':
    # Real command
    # main(sys.argv)
    main(['ex2.py', 'train_x.txt', 'train_y.txt', 'test_x.txt'])


# Checks
train_data = load_data('train_x.txt')
print('train_data', train_data)
train_labels = load_labels('train_y.txt')
print('train_labels', train_labels)
train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
print('train_x', train_x, len(train_x))
print('train_y', train_y, len(train_y))
print('dev_x', dev_x, len(dev_x))
print('dev_y', dev_y, len(dev_y))
