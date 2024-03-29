import scipy.io
import numpy as np
import single_layer as sl
from sklearn import preprocessing
import time
import multi_layer as ml


if __name__ == '__main__':
    train = scipy.io.loadmat('data/train.mat')
    train_images = train['train']['images'][0][0] # (28, 28, 60000) ndarray
    train_labels = train['train']['labels'][0][0] # (60000, 1) ndarray
    # train = scipy.io.loadmat('data/train_small.mat')
    # train_images = train['train'][0][6][0][0][0]
    # train_labels = train['train'][0][6][0][0][1]
    # (60000, 784) ndarray
    train_images = \
        np.reshape(np.transpose(train_images, [2, 0, 1]), (60000, 784))

    train_images = preprocessing.scale(train_images.astype(float), axis=1)

    test = scipy.io.loadmat('data/test.mat')
    test_images = test['test']['images'][0][0]
    test_labels = test['test']['labels'][0][0]
    test_images = \
        np.reshape(np.transpose(test_images, [2, 0, 1]), (10000, 784))

    test_images = preprocessing.scale(test_images.astype(float), axis=1)

    # start = time.clock()
    # sl.run_epoches(train_images, train_labels, test_images, test_labels)
    # end = time.clock()
    # elapsed = (end - start) / 3600
    # print 'elapsed time (hours) for single layered network=', elapsed

    start = time.clock()
    ml.run_epoches(train_images, train_labels, test_images, test_labels)
    end = time.clock()
    elapsed = (end - start) / 3600
    print 'elapsed time (hours) for single layered network=', elapsed
