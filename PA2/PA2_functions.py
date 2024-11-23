import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(x))


# code from PA1 - assume input dim is 2
def softmax(input :np.ndarray):
    input = input - np.max(input, axis=1, keepdims=True)
    output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
    return output


# code from PA1 - assume label is one-hot vector
def cross_entropy_loss(x_softmax:np.ndarray, label:np.ndarray):
    batch_size = x_softmax.shape[0]
    return -np.sum( label * np.log(x_softmax + 1e-7 ) ) / batch_size