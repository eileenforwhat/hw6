import numpy as np
import helper
import matplotlib.pyplot as plt
import math


def run_epoches(images, labels, n=200, alpha=0.6):
    """Run n number of epoches."""

    # initialize weights and bias values to random
    scale_factor = 10e-2 # scaling factor for weight and bias
    # weights and bias for Mean Squared Error
    mse_weights = [np.random.randn(784, 300) * scale_factor, \
            np.random.randn(300, 100) * scale_factor, \
            np.random.randn(100, 10) * scale_factor]
    mse_bias = [np.random.randn(300, 1) * scale_factor,\
            np.random.randn(100, 1) * scale_factor, \
            np.random.randn(10, 1) * scale_factor]

    # weights and bias for Cross-Entropy Error
    cee_weights = [np.random.randn(784, 300) * scale_factor, \
            np.random.randn(300, 100) * scale_factor, \
            np.random.randn(100, 10) * scale_factor]
    cee_bias = [np.random.randn(300, 1) * scale_factor , \
            np.random.randn(100, 1) *scale_factor, \
            np.random.randn(10, 1) *scale_factor]

    for i in range(n):
        eta = alpha / math.pow(i + 1, 0.5)
        batches = helper.generate_batches(images, labels)
        for batch in batches:       

            # extract features out of the training set
            feats = batch[:,0:784] # each row is a feature
            x_mse = forward(feats, mse_weights, mse_bias)
            x_cee = forward(feats, cee_weights, cee_bias)

    return (mse_weights, mse_bias, cee_weights, cee_bias)


def forward(x_0, weights, bias):
    """
     calculates the hidden layers x_1, x_2 with tanh function
     which is defined to be
     tanh(s) = (exp(s) - exp(-s)) / (exp(s) + exp(-s))

     then calculate the last layer x_3 with sigmoid function
    """
    # sigmoid function
    def sigmoid(s):
        return 1.0 / (1 + np.exp(-s))
    # sigmoid function to accept numpy array
    sigmoid = np.vectorize(sigmoid, otypes=[np.float])

    s_1 = np.dot(x_0, weights[0]) + bias[0].T # 200-by-300 matrix
    x_1 = np.tanh(s_1)

    s_2 = np.dot(x_1, weights[1]) + bias[1].T # 200-by-100 matrix
    x_2 = np.tanh(s_2)

    s_3 = np.dot(x_2, weights[2]) + bias[2].T # 200-by-10 matrix
    x_3 = sigmoid(s_3)
    return [x_0, x_1, x_2, x_3]

def backward(x_3, weights, bias):
    
