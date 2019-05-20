import sys
import numpy as np


def load_data(data_file):
    '''
    Load the features from .txt file
    Convert the 'sex' feature to an index.
    Add a bias term to the feature vector.
    '''
    data = []
    # Sex feature to numerical index dictionary
    sex_to_index = {'M': 0, 'F': 1, 'I': 2}
    # Read file
    with open(data_file, 'r') as file:
        for line in file:
            # Split entry
            line = line.strip().split(',')
            line[0] = sex_to_index[line[0]]
            # Add bias
            line.append(1)
            # Add to data
            data.append(np.array(line, dtype=np.float64))
    return np.array(data)


def load_labels(labels_file):
    '''
    Read labels from .txt file
    '''
    # Read file
    labels = np.loadtxt(labels_file, dtype=np.float64)
    return labels


def normalize_data(data, mean, std):
    '''
    Apply z-score normalization, given train set mean and standard deviation
    '''
    # Iterate all features except the bias term.
    for i in range(len(std) - 1):
        if std[i] == 0:
            # Do not divide std if it is 0
            data[:, i] = data[:, i] - mean[i]
        else:
            # z-score
            data[:, i] = (data[:, i] - mean[i]) / std[i]
    return data


def train_dev_split(train_data, train_labels):
    '''
    Split the original train file to 80% train and 20% validation data.
    '''
    # Number of samples
    N = len(train_data)
    # Shuffle data and labels in the same way
    arr = np.arange(N)
    np.random.seed(seed=42)
    np.random.shuffle(arr)
    train_data = train_data[arr]
    train_labels = train_labels[arr]
    # Split the shuffled data to 80% train and 20% validation
    stop_index = int(N*0.8)
    train_x = train_data[:stop_index]
    train_y = train_labels[:stop_index]
    dev_x = train_data[stop_index:]
    dev_y = train_labels[stop_index:]
    return train_x, dev_x, train_y, dev_y


def train(train_x, train_y, epochs, eta, Lambda, key):
    '''
    Training algorithms
    '''
    # Number of samples
    N = len(train_x)
    # Number of original features plus one dummy feature
    n = len(train_x[0])
    # Number of classes
    k = 3
    # Initialize parameters to 0.
    w = np.zeros((k, n))
    # Epochs
    ep = 0
    # Initialize validation accuracy
    dev_acc = 0
    # Iterate over epochs (including early stopping condition)
    while ep < epochs and dev_acc < 0.65:
        # Shuffle train_x and train_y the same way
        arr = np.arange(N)
        np.random.seed(seed=42)
        np.random.shuffle(arr)
        train_x = train_x[arr]
        train_y = train_y[arr]
        # Eta decay
        if key == 'per' and ep % 5 == 0:
            eta *= 0.8
        elif key == 'svm' and ep % 10 == 0:
            eta *= 0.8
        elif key == 'pa':
            pass
        # Iterate over training samples
        for i in range(N):
            # Get sample data and label
            x = train_x[i]
            y = train_y[i]
            y = int(y)
            # Calculate the class scores
            values = np.dot(w, x)
            # Update rules
            # Perceptron
            if key == 'per':
                # Find argmax
                y_hat = np.argmax(values)
                # If the prediction doesn't match, update w
                if y_hat != y:
                    w[y, :] = w[y, :] + eta * x
                    w[y_hat, :] = w[y_hat, :] - eta * x
            # SVM
            elif key == 'svm':
                # Set 'y' index to -inf to exclude it from the argmax
                values[y] = - np.inf
                # Find argmax (without y)
                y_hat = np.argmax(values)
                s = 1 - eta * Lambda
                for l in range(k):
                    if l == y:
                        w[l, :] = s * w[l, :] + eta * x
                    elif l == y_hat:
                        w[l, :] = s * w[l, :] - eta * x
                    else:
                        w[l, :] = s * w[l, :]
            # PA
            elif key == 'pa':
                # Set 'y' index to -inf to exclude it from the argmax
                values[y] = - np.inf
                # Find argmax (without y)
                y_hat = np.argmax(values)
                # Compute tau
                err = 1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x)
                loss = np.max([0, err])
                tau = loss/(2*np.dot(x,x))
                # Update w
                for l in range(k):
                    if l == y:
                         w[y, :] = w[y, :] + tau * x
                    if l == y_hat:
                        w[y_hat, :] = w[y_hat, :] - tau * x
        # Compute validation accuracy
        dev_acc = evaluate(dev_x, dev_y, w)
        print(dev_acc)
        # Update epoch index
        ep += 1
    print()
    return w


def evaluate(dev_x, dev_y, w):
    '''
    Compute validation set accuracy using trained parameters
    '''
    # Initialization
    accuracy = 0
    # Number of samples
    N = len(dev_x)
    for x, y in zip(dev_x, dev_y):
        # Calculate the class scores
        values = np.dot(w, x)
        # Find argmax
        y_hat = np.argmax(values)
        # Correct classification
        if y_hat == y:
            accuracy += 1
    # Divide by number of samples
    accuracy /= N
    return accuracy


def predict(test_x, w):
    '''
    Predict labels for the test set
    '''
    # Number of samples
    N = len(test_x)
    # Labels initialization
    y_hats = np.zeros(N, dtype=int)
    for i in range(N):
        # Get sample features
        x = test_x[i]
        # Calculate the class scores
        values = np.dot(w, x)
        # Find argmax
        y_hat = np.argmax(values)
        # Set prediction
        y_hats[i] = y_hat
    return y_hats


def main(argv):
    # Load the data from files
    train_data = load_data(argv[1])
    # Compute mean and standard deviation
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    # Normalize train data
    train_data = normalize_data(train_data, mean, std)
    # Load labels from file
    train_labels = load_labels(argv[2])
    # Load the data from files
    test_x = load_data(argv[3])
    # Normalize test data
    test_x = normalize_data(test_x, mean, std)
    # Make validation data global (for evaluation)
    global dev_x, dev_y
    # Split original train to train and validation sets
    train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
    # Train parameters with 3 algorithms
    # Perceptron
    w_per = train(train_x, train_y, epochs=150, eta=0.01, Lambda=None,  key='per')
    # SVM
    w_svm = train(train_x, train_y, epochs=150, eta=0.1, Lambda=0.1,  key='svm')
    # PA
    w_pa = train(train_x, train_y, epochs=100, eta=None, Lambda=None,  key='pa')
    # Predict test and print prediction
    y_hats_per = predict(test_x, w_per)
    y_hats_svm = predict(test_x, w_svm)
    y_hats_pa = predict(test_x, w_pa)
    # Print all predictions in required format
    for i in range(len(test_x)):
        print('perceptron: ' + str(y_hats_per[i]) + ', ' + 'svm: ' +
              str(y_hats_svm[i]) + ', ' 'pa: ' + str(y_hats_pa[i]))


# Main
if __name__ == '__main__':
    main(sys.argv)
