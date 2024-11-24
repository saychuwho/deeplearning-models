import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(x))


# code from PA1 & modified to accept input dim is 1
def softmax(input :np.ndarray):
    input = input - np.max(input)
    output = np.exp(input) / np.sum(np.exp(input))
    return output


# code from PA1 - assume label is one-hot vector
def cross_entropy_loss(x_softmax:np.ndarray, label:np.ndarray):
    batch_size = x_softmax.shape[0]
    return -np.sum( label * np.log(x_softmax + 1e-7 ) ) / batch_size