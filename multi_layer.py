import numpy as np
import helper
import matplotlib.pyplot as plt
import math


def run_epoches(train_images, train_labels, test_images, test_labels, n=200, alpha=0.6):
    """
    Run n number of epoches.
    train_images : training images
    train_lables : training labels
    test_images : test images
    test_labels : test labels
    n : number of epoches
    alpha : initial learning rate 
    """

    # initialize weights and bias values to random
    scale_factor = 10e-4 # scaling factor for weight and bias
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
    
    # list of accuracies for different loss function
    training_mse = []
    training_cee = []
    test_mse = []
    test_cee = []
    x_axis = []

    for i in range(n):
        print i,'-th epoch'
        eta = alpha / math.pow(i + 1, 0.5)
        batches = helper.generate_batches(train_images, train_labels)
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

            res1 = calculate_accuracy(predict(train_images, mse_weights, mse_bias), train_labels)
            res2 = calculate_accuracy(predict(train_images, cee_weights, cee_bias), train_labels)
            res3 = calculate_accuracy(predict(test_images, mse_weights, mse_bias), test_labels)
            res4 = calculate_accuracy(predict(test_images, cee_weights, cee_bias), test_labels)

            training_mse.append(res1)
            training_cee.append(res2)
            test_mse.append(res3)
            test_cee.append(res4)

            print 'epoch=', i
            print 'error rate on training set using mean squared error', 1 - res1
            print 'error rate on training set using cross-entropy error', 1- res2
            print 'error rate on test set using mean squared error', 1-res3
            print 'error rate on test set using cross-entropy error', 1-res4

    p1, = plt.plot(x_axis, training_mse, 'r')
    p2, = plt.plot(x_axis, training_cee, 'b')
    p3, = plt.plot(x_axis, test_mse, 'g')
    p4, = plt.plot(x_axis, test_cee, 'k')
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
    def get_sigmoid(s):
        ret = 1.0 / (1 + np.exp(-s))
        offset = 10e-5
        if ret == 0:
            ret += offset # 0.0001
        elif ret == 1:
            ret -= offset # 0.9999
        return ret
    def get_tanh(s):
        ret = np.tanh(s)
        offset = 10e-5
        if ret==-1:
            ret+=offset
        elif ret==0:
            ret+=offset
        elif ret==1:
            ret-=offset
        return ret
    # sigmoid and tanh to accept numpy array
    get_sigmoid = np.vectorize(get_sigmoid, otypes=[np.float])
    get_tanh = np.vectorize(get_tanh, otypes=[np.float])

    s_1 = np.dot(x_0, weights[0]) + bias[0].T # 200-by-300 matrix
    x_1 = get_tanh(s_1)

    s_2 = np.dot(x_1, weights[1]) + bias[1].T # 200-by-100 matrix
    x_2 = get_tanh(s_2)

    s_3 = np.dot(x_2, weights[2]) + bias[2].T # 200-by-10 matrix
    x_3 = get_sigmoid(s_3)
    return [x_0, x_1, x_2, x_3]


def backward_mse(x, labels, weights, bias):
    x_0, x_1, x_2, x_3 = x
    t = np.zeros((200, 10))
    for row in range(len(t[:,0])):
        r = np.zeros((1, 10))
        r[:,int(labels[row])] = 1
        t[row,:] = r
    d_3 = np.multiply(np.multiply(x_3 - t, 1 - x_3), x_3) # 200-by-10
    d_2 = np.multiply(np.dot(d_3, weights[2].T), 1 - np.multiply(x_2, x_2)) # 200-by-100
    d_1 = np.multiply(np.dot(d_2, weights[1].T), 1 - np.multiply(x_1, x_1)) # 200-by-300
    return [d_1, d_2, d_3]


def backward_cee(x, labels, weights, bias):
    x_0, x_1, x_2, x_3 = x
    t = np.zeros((200, 10))
    for row in range(len(t[:,0])):
        r = np.zeros((1, 10))
        r[:,int(labels[row])] = 1
        t[row,:] = r
    d_3 = x_3 - t # 200-by-10
    d_2 = np.multiply(np.dot(d_3, weights[2].T), 1 - np.multiply(x_2, x_2)) # 200-by-100
    d_1 = np.multiply(np.dot(d_2, weights[1].T), 1 - np.multiply(x_1, x_1)) # 200-by-300
    return [d_1, d_2, d_3]


def update_w(x, weights, deltas, eta):
    for i in range(3):
        weights[i] = weights[i] - eta * np.dot(x[i].T, deltas[i])
    return weights


def update_b(x, biases, deltas, eta):
    for i in range(3):
        # sum the deltas
        delta_sum = np.sum(deltas[i], axis=0)
        # reshaping so that it matches the shape of biases
        delta_sum = delta_sum.reshape(delta_sum.shape[0], 1)
        biases[i] = biases[i] - eta * delta_sum
    return biases

def predict(features, weights, biases):
    mse_pred = forward(features, weights, biases) # propagate
    mse_pred = np.argmax(mse_pred[3],axis=1) # classify
    return mse_pred.reshape(mse_pred.shape[0], 1)

def calculate_accuracy(pred, true_label):
    return 1.0 * np.sum(pred==true_label) / true_label.shape[0]

def calculate_error(pred, true_label):
    return 1.0 * np.sum(pred!=true_label) / true_label.shape[0]
