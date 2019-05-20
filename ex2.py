import sys
import numpy as np
from random import shuffle


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
            line.append(1)
            # Add to data
            data.append(np.array(line, dtype=np.float64))
    return np.array(data)


def load_labels(labels_file):
    # Read labels from file
    labels = np.loadtxt(labels_file, dtype=np.float64)
    return labels


def normalize_data(data, mean, std):
    for i in range(len(std) - 1):
        if std[i] == 0:
            data[i] = data[i] - mean[i]
        else:
            data[i] = (data[i] - mean[i]) / std[i]
    return data


def train_dev_split(train_data, train_labels):
    # We split the original train file to 80% train and 20% validation data
    # We will delete this after hyper-parameters tuning and use only test file
    N = len(train_data)
    arr = np.arange(N)
    np.random.seed(seed=42)
    np.random.shuffle(arr)
    train_data = train_data[arr]
    train_labels = train_labels[arr]
    stop_index = int(N*0.8)
    train_x = train_data[:stop_index]
    train_y = train_labels[:stop_index]
    dev_x = train_data[stop_index:]
    dev_y = train_labels[stop_index:]
    return train_x, dev_x, train_y, dev_y


def train(train_x, train_y, epochs, eta, Lambda, key):
    N = len(train_x)
    n = len(train_x[0]) # number of original features plus one dummy feature
    k = 3
    # initialize parameters to 0.
    w = np.zeros((k, n))
    ep = 0
    dev_acc = 0
    while ep < epochs and dev_acc < 0.65:
        # shuffle train_x and train_y the same way
        arr = np.arange(N)
        np.random.seed(seed=42)
        np.random.shuffle(arr)
        train_x = train_x[arr]
        train_y = train_y[arr]
        if key == 'per' and ep % 5 == 0:
            eta *= 0.8
        elif key == 'svm' and ep % 10 == 0:
            eta *= 0.8
        elif key == 'pa':
            pass
        for i in range(N):
            x = train_x[i]
            y = train_y[i]
            y = int(y)
            values = np.dot(w, x)
            if key == 'per':
                y_hat = np.argmax(values)
                # if the prediction doesn't match, update w
                if y_hat != y:
                    w[y, :] = w[y, :] + eta * x
                    w[y_hat, :] = w[y_hat, :] - eta * x
            elif key == 'svm':
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
            elif key == 'pa':
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
        dev_acc = evaluate(dev_x, dev_y, w)
        ep += 1
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


def predict(test_x, w):
    N = len(test_x)
    y_hats = np.zeros(N, dtype=int)
    for i in range(N):
        x = test_x[i]
        values = np.dot(w, x)
        y_hat = np.argmax(values)
        y_hats[i] = y_hat
    return y_hats


def main(argv):
    train_data = load_data(argv[1])
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = normalize_data(train_data, mean, std)
    train_labels = load_labels(argv[2])
    test_x = load_data(argv[3])
    test_x = normalize_data(test_x, mean, std)
    global dev_x, dev_y
    train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
    # Train parameters with 3 algorithms
    w_per = train(train_x, train_y, epochs=150, eta=0.01, Lambda=None,  key='per')
    w_svm = train(train_x, train_y, epochs=150, eta=0.1, Lambda=0.1,  key='svm')
    w_pa = train(train_x, train_y, epochs=100, eta=None, Lambda=None,  key='pa')
    # Predict test and print prediction
    y_hats_per = predict(test_x, w_per)
    y_hats_svm = predict(test_x, w_svm)
    y_hats_pa = predict(test_x, w_pa)
    for i in range(len(test_x)):
        print('perceptron: ' + str(y_hats_per[i]) + ', ' + 'svm: ' +
              str(y_hats_svm[i]) + ', ' 'pa: ' + str(y_hats_pa[i]))


# main
if __name__ == '__main__':
    main(sys.argv)
    # main(['ex2.py', 'train_x.txt', 'train_y.txt', 'train_x.txt'])
