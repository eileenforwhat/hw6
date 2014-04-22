import numpy as np
import helper
import matplotlib.pyplot as plt
import math


def run_epoches(images, labels, test_images, test_labels, n=200, alpha=0.01):
    """Run n number of epoches."""
    mse_weights = np.random.rand(784, 10) - 0.5
    mse_bias = np.random.rand(10, 1)
    entropy_weights = np.random.rand(784, 10) - 0.5
    entropy_bias = np.random.rand(10, 1)

    # for plot
    x_axis = []
    training_mse = []
    training_entropy = []
    test_mse = []
    test_entropy = []

    for i in range(n):
        eta = alpha / math.pow(i + 1, 0.5)
        batches = helper.generate_batches(images, labels)
        for batch in batches:
            gradient_mse_w = compute_gradient_mse(batch, mse_weights, mse_bias, 'w')
            gradient_mse_b = compute_gradient_mse(batch, mse_weights, mse_bias, 'b')
            gradient_entropy_w = \
                compute_gradient_entropy(batch, entropy_weights, entropy_bias, 'w')
            gradient_entropy_b = \
                compute_gradient_entropy(batch, entropy_weights, entropy_bias, 'b')
            mse_weights = mse_weights + eta * gradient_mse_w
            mse_bias = mse_bias + eta * gradient_mse_b
            entropy_weights = entropy_weights + eta * gradient_entropy_w
            entropy_bias = entropy_bias + eta * gradient_entropy_b

        #storing info every 10 epochs
        if i % 10 == 0:
            x_axis.append(i)
            y1 = helper.error(mse_weights, mse_bias, images, labels)
            training_mse.append(1 - y1)
            y2 = helper.error(entropy_weights, entropy_bias, images, labels)
            training_entropy.append(1 - y2)
            y3 = helper.error(mse_weights, mse_bias, test_images, test_labels)
            test_mse.append(1 - y3)
            y4 = helper.error(entropy_weights, entropy_bias,
                    test_images, test_labels)
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
    return (mse_weights, mse_bias, entropy_weights, entropy_bias)


def compute_gradient_mse(batch, weights, bias, b_or_w):
    """
    Computes the mean squared error gradient by summing over gradients
    of all data points in batch.
    """
    if b_or_w == 'w':
        ret = np.zeros((784, 10))
    else:
        ret = np.zeros((10, 1))

    for dp in batch:
        x = dp.T[:784].reshape(784, 1)
        t = np.zeros((10, 1))
        t[int(dp.T[784:][0])] = 1
        y = helper.sigmoid(x, weights, bias)
        if b_or_w == 'w':
            v = np.diagonal(np.dot(np.diagonal(np.dot((y - t), (1 - y).T)).\
                reshape(10, 1), y.T)).reshape(10, 1)
            ret += np.dot(x, v.T)
            # addition = np.zeros((784, 10))
            # col = 0
            # for v_k in v:
            #   addition[:,col] = (v_k[0] * x).reshape(784)
            #   col += 1
            # ret += addition
        else: # b_or_w == 'b':
            ret += np.diagonal(np.dot(np.diagonal(np.dot((y - t), (1 - y).T)).\
                reshape(10, 1), y.T)).reshape(10, 1)
    return ret


def compute_gradient_entropy(batch, weights, bias, b_or_w):
    """
    Computes the cross-entropy error gradient by summing over gradients
    of all data points in batch.
    """
    if b_or_w == 'w':
        ret = np.zeros((784, 10))
    else:
        ret = np.zeros((10, 1))
    for dp in batch:
        x = dp.T[:784].reshape(784, 1)
        t = np.zeros((10, 1))
        t[int(dp.T[784:][0])] = 1
        y = helper.sigmoid(x, weights, bias)
        if b_or_w == 'w':
            v = y - t
            addition = np.zeros((784, 10))
            col = 0
            for v_k in v:
                addition[:,col] = (v_k[0] * x).reshape(784)
                col += 1
            ret += addition
        else: # b_or_w == 'b':
            ret += y - t
    return ret
