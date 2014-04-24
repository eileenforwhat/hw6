import matplotlib.pyplot as plt


if __name__ == '__main__':
    x_axis = [0, 10, 20, 30]
    print 1
    training_mse = [0.09930000000000005, 0.09863333333299995, 0.09863333333299995, 0.0986334]
    print 2
    training_entropy = [0.34568333333300005, 0.704616666667, 0.70245, 0.75225]
    print 3
    test_mse = [0.10319999999999996, 0.0958, 0.0958, 0.0958]
    print 4
    test_entropy = [0.34650000000000003, 0.7113, 0.7078,0.7561]
    print 5
    p1, = plt.plot(x_axis, training_mse, 'r')
    print 6
    p2, = plt.plot(x_axis, training_entropy, 'b')
    print 7
    p3, = plt.plot(x_axis, test_mse, 'g')
    print 8
    p4, = plt.plot(x_axis, test_entropy, 'k')
    print 9
    plt.legend([p1, p2, p3, p4],
        ['training accuracy, mse', 'training accuracy, entropy',
            'test accuracy, mse', 'test accuracy, entropy'])
    print 10
    plt.show()  
    print 11 
