import scipy.io
import numpy as np
from single_layer import run_epoches
import time
from sklearn import preprocessing
import multi_layer as ml


if __name__ == '__main__':
    train = scipy.io.loadmat('data/train.mat')
    train_images = train['train']['images'][0][0] # (28, 28, 60000) ndarray
    train_labels = train['train']['labels'][0][0] # (60000, 1) ndarray
    # (60000, 784) ndarray
    train_images = \
        np.reshape(np.transpose(train_images, [2, 0, 1]), (60000, 784))
    train_images = preprocessing.scale(train_images.astype(float))

    test = scipy.io.loadmat('data/test.mat')
    test_images = test['test']['images'][0][0]
    test_labels = test['test']['labels'][0][0]
    test_images = \
        np.reshape(np.transpose(test_images, [2, 0, 1]), (10000, 784))
    test_images = preprocessing.scale(test_images.astype(float))

#    start = time.clock()
#    mse_weights, mse_bias, entropy_weights, entropy_bias = \
#        run_epoches(train_images, train_labels, test_images, test_labels, n=300)
#    end = time.clock()
#    elapsed = (end - start) / 3600
#    print 'elapsed time (hours) for single layered network=', elapsed
    mse_weights, mse_bias, entropy_weights, entropy_bias = \
        ml.run_epoches(train_images, train_labels)
