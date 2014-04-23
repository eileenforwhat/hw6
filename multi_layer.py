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
            feats = batch[:,:784] # each row is a feature
            labels = batch[:,784:] # each row is a label

            x_mse = forward(feats, mse_weights, mse_bias)
            x_cee = forward(feats, cee_weights, cee_bias)

            # [d_1, d_2, d_3]
            d_mse = backward_mse(x_mse, labels, mse_weights, mse_bias)
            d_cee = backward_cee(x_cee, labels, cee_weights, cee_bias)

            mse_weights = update_w(x_mse, mse_weights, d_mse, eta)
            cee_weights = update_w(x_cee, cee_weights, d_cee, eta)
            mse_bias = update_b(x_mse, mse_bias, d_mse, eta)
            cee_bias = update_b(x_cee, cee_bias, d_cee, eta)

        #storing info every 10 epochs
        if i % 10 == 0:

            x_axis.append(i)
            y1 = helper.error(mse_weights, mse_bias, images, labels)
            training_mse.append(1 - y1)
            
            y2 = helper.error(cee_weights, cee_bias, images, labels)
            training_entropy.append(1 - y2)
           
            y3 = helper.error(mse_weights, mse_bias, test_images, test_labels)
            test_mse.append(1 - y3)
            
            y4 = helper.error(cee_weights, cee_bias, test_images, test_labels)
            test_entropy.append(1 - y4)

            print 'epoch=', i
            print 'error rate on training set using mean squared error', y1
            print 'error rate on training set using cross-entropy error', y2
            print 'error rate on test set using mean squared error', y3
            print 'error rate on test set using cross-entropy error', y4

    p1, = plt.plot(x_axis, training_mse, 'r')
    p2, = plt.plot(x_axis, training_entropy, 'b')
    p3, = plt.plot(x_axis, test_mse, 'g')
    p4, = plt.plot(x_axis, test_entropy, 'k')
    plt.legend([p1, p2, p3, p4],
        ['training accuracy, mse', 'training accuracy, entropy',
            'test accuracy, mse', 'test accuracy, entropy'])
    plt.show()


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

def backward_mse(x, labels, weights, bias):
    x_0, x_1, x_2, x_3 = x
    t = np.zeros((200, 10))
    for row in range(len(t[:,0])):
        r = np.zeros((1, 10))
        r[:,int(labels[row])] = 1
        t[row,:] = r
    d_3 = np.multiply(np.multiply(x_3 - t, 1 - x_3), x_3)
    d_2 = np.multiply(np.dot(d_3, weights[2].T), 1 - np.multiply(x_2, x_2))
    d_1 = np.multiply(np.dot(d_2, weights[1].T), 1 - np.multiply(x_1, x_1))
    return [d_1, d_2, d_3]


def backward_cee(x, labels, weights, bias):
    x_0, x_1, x_2, x_3 = x
    t = np.zeros((200, 10))
    for row in range(len(t[:,0])):
        r = np.zeros((1, 10))
        r[:,int(labels[row])] = 1
        t[row,:] = r
    d_3 = x_3 - t
    d_2 = np.multiply(np.dot(d_3, weights[2].T), 1 - np.multiply(x_2, x_2))
    d_1 = np.multiply(np.dot(d_2, weights[1].T), 1 - np.multiply(x_1, x_1))
    return [d_1, d_2, d_3]


def update_w(x, weights, deltas, eta):
    for i in range(3):
        weights[i] = weights[i] - eta * np.dot(x[i].T, deltas[i+1])
    return weights


def update_b(x, weights, deltas, eta):
    for i in range(3):
        weights[i] = weights[i] - eta * deltas[i+1]
    return weights
