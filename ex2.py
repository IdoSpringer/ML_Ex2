import sys
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
# CANNOT USE SKLEARN... FIX LATER (or delete train test split)

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


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if any(int(value) for value in std) == 0:
        # normalize feature in different way
        pass
    # (notice - do not divide by zero)
    data = (data - mean) / std
    return data


def train_dev_split(train_data, train_labels):
    # We split the original train file to 80% train and 20% validation data
    # We will delete this after hyper-parameters tuning and use only test file
    train_x, dev_x, train_y, dev_y = train_test_split(train_data, train_labels, test_size=0.2)
    return train_x, dev_x, train_y, dev_y


def train_perceptron(train_x, train_y, epochs, eta, k):
    # Perceptron Algorithm
    N = len(train_x)  # number of data points
    n = len(train_x[0])  # number of features
    # w represents the coefficients of x, b represents the free coefficient
    # since we implement multiclass perceptron with 3 classes, we have 3 sets of w and b
    # initialize parameters to 0.
    w = np.zeros((k, n))
    # b = np.zeros(k)
    for ep in range(epochs):
        # shuffle train_x and train_y the same way
        arr = np.arange(N)
        np.random.shuffle(arr)
        train_x = train_x[arr]
        train_y = train_y[arr]
        for i in range(N):
            # find y hat - the predicted class - as the argmax of the products
            x = train_x[i]
            y = train_y[i]
            values = np.dot(w, x) # + b
            y_hat = np.argmax(values)
            # if the prediction doesn't match, update w,b
            if y_hat != y:
                y = int(y)
                w[y, :] = w[y, :] + eta * x
                # b[y] = b[y] + eta
                w[y_hat, :] = w[y_hat, :] - eta * x
                # b[y_hat] = b[y_hat] - eta
        print(evaluate(dev_x, dev_y, w))
    return w


def train_svm(train_x, train_y, epochs, eta, Lambda, k):
    # Support Vector Machine algorithm
    N = len(train_x)
    n = len(train_x[0])
    w = np.zeros((k, n))
    for ep in range(epochs):
        arr = np.arange(N)
        np.random.shuffle(arr)
        train_x = train_x[arr]
        train_y = train_y[arr]
        for i in range(N):
            x = train_x[i]
            y = train_y[i]
            y = int(y)
            values = np.dot(w, x)
            # Put -inf in y index to find argmax without y
            values[y] = - np.inf
            y_hat = np.argmax(values)
            s = 1 - eta * Lambda
            for l in range(k):
                if l == y:
                    w[l, :] = s * w[l, :] + eta * x
                elif l == y_hat:
                    w[l, :] = s * w[l, :] - eta * x
                else:
                    w[l, :] = s * w[l, :]
        print(evaluate(dev_x, dev_y, w))
    return w


def train_pa(train_x, train_y, epochs, k):
    # Passive Aggressive algorithm
    N = len(train_x)
    n = len(train_x[0])
    # initialize parameters to 0.
    w = np.zeros((k, n))
    for ep in range(epochs):
        #shuffle train_x and train_y the same way
        arr = np.arange(N)
        np.random.shuffle(arr)
        train_x = train_x[arr]
        train_y = train_y[arr]
        for i in range(N):
            # find y hat - the predicted class - as the argmax of the products, without y
            x = train_x[i]
            y = train_y[i]
            y = int(y)
            values = np.dot(w, x)
            # put -inf in y index to find argmax without y
            values[y] = - np.inf
            y_hat = np.argmax(values)
            # compute tau
            err = 1-np.dot(w[y, :], x)+np.dot(w[y_hat, :], x)
            loss = np.max([0, err])
            tau = loss/np.dot(x,x)
            # update w
            for l in range(k):
                if l == y:
                     w[y, :] = w[y, :] + tau * x
                if l == y_hat:
                    w[y_hat, :] = w[y_hat, :] - tau * x
        print(evaluate(dev_x, dev_y, w))
    return w


def evaluate(dev_x, dev_y, w):
    # compute dev accuracy using trained parameters
    accuracy = 0
    N = len(dev_x)
    for x, y in zip(dev_x, dev_y):
        values = np.dot(w, x)
        y_hat = np.argmax(values)
        if y_hat == y:
            accuracy += 1
    accuracy /= N
    return accuracy


def predict(test_x, parameters):
    pass


def main(argv):
    train_data = load_data(argv[1])
    train_labels = load_labels(argv[2])
    global dev_x, dev_y
    train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
    train_data = normalize_data(train_data)
    # Train parameters with 3 algorithms
    train_perceptron(train_x, train_y, epochs=100, eta=0.01, k=3)
    train_svm(train_x, train_y, epochs=100, eta=0.01, Lambda=0.5,  k=3)
    # Predict test and print prediction
    pass


# main
if __name__ == '__main__':
    # Real command
    # main(sys.argv)
    main(['ex2.py', 'train_x.txt', 'train_y.txt', 'test_x.txt'])


# Checks
'''
train_data = load_data('train_x.txt')
print('train_data', train_data)
train_labels = load_labels('train_y.txt')
print('train_labels', train_labels)
train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
print('train_x', train_x, len(train_x))
print('train_y', train_y, len(train_y))
print('dev_x', dev_x, len(dev_x))
print('dev_y', dev_y, len(dev_y))
'''
