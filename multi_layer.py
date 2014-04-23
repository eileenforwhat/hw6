import numpy as np
import helper
import matplotlib.pyplot as plt


def run_epoches(images, labels, n=200, eta=0.1):
    """Run n number of epoches."""
    # initialize weights and bias values to random
    mse_weights = [np.random.rand(784, 300), np.random.rand(300, 100), \
        np.random.rand(100, 10)]
    mse_bias = [np.random.rand(300, 1), np.random.rand(100, 1), \
        np.random.rand(10, 1)]
    #entropy_weights = [np.random.rand(784, 300), np.random.rand(784, 100), \
    #    np.random.rand(784, 10)]
    entropy_weights = [np.random.rand(784, 300), np.random.rand(300, 100), \
        np.random.rand(100, 10)]
    entropy_bias = [np.random.rand(300, 1), np.random.rand(100, 1), \
        np.random.rand(10, 1)]
    for i in range(n):
        batches = helper.generate_batches(images, labels)
        for batch in batches:       

            # extract features out of the training set
            feats = batch[:,0:784] # each row is a feature
            x_layers = forward(feats, mse_weights, mse_bias)


#            # x_mse_w : new x value wrt w and b and mse
#            x_mse_w = forward(images, mse_weights, mse_bias)
#            sigma_mse_w = backward(x_mse_w, mse_weights, mse_bias)
#
#            x_entropy_w = forward(images, entropy_weights, entropy_bias)
#            sigma_entropy_w = backward(x_entropy_w, entropy_weights, entropy_bias)
#
#            x_mse_b = forward(images, mse_weights, mse_bias)
#            sigma_mse_b = backward(x_mse_b, mse_weights, mse_bias)
#            
#            x_entropy_b = forward(images ,entropy_weights, entropy_bias)
#            sigma_entropy_b = backward(x_entropy_b, entropy_weights, entropy_bias)
#            
#            mse_weights = update(x_mse_w, sigma_mse_w, eta)
#            mse_bias = update(x_mse_b, sigma_mse_b, eta)
#            entropy_weights = update(x_entropy_w, sigma_entropy_w, eta)
#            entropy_bias = update(x_entropy_b, sigma_entropy_b, eta)
    return (mse_weights, mse_bias, entropy_weights, entropy_bias)

"""
 calculates the hidden layers x_1, x_2 with tanh function
 which is defined to be
 tanh(s) = (exp(s) - exp(-s)) / (exp(s) + exp(-s))

 then calculate the last layer x_3 with sigmoid function
"""
def forward(x_0, weights, bias):
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
    
    #x_1 = np.tanh(np.dot(x_0, weights[0]))
    #print x_1
    #x_2 = np.tanh(np.dot(x_1, weights[1]))
    #print x_2
    #x_3 = np.tanh(np.dot(x_2, weights[2]))
    #x_1 = np.tanh(np.dot(weights[0].T, x_0.T).T)
    #x_2 = np.tanh(np.dot(weights[1].T, x_1.T).T)
    #x_3 = np.tanh(np.dot(weights[2].T, x_2.T).T)
    #def tanh(x,w,b):
    #    x = x.reshape(x.shape[0],1)
    #    #x = np.matrix(x).T
    #    s = np.dot(w.T,x) + b
    #    s[s < 10e-7] = 0 # for numerical issue
    #    tanh_numer = np.exp(s) - np.exp(-s)
    #    tanh_denom = np.exp(s) + np.exp(-s)
    #    # # # Replace NaN to zero. 
    #    # # # Not sure if I am allowed to?
    #    return np.nan_to_num(tanh_numer / tanh_denom)
    #x_1 = np.zeros((300, 1))
    #for feat in x_0:
    #    x_1 += tanh(feat, weights[0], bias[0])
    #x_1 = tanh(feat, weights[0], bias[0])
    #x_2 = tanh(x_1, weights[1], bias[1])
    #x_3 = helper.sigmoid(x_2, weights[2], bias[2]) # last layer
    #x_1 = step_forward(x_0, weights[0], bias[0])
    #x_2 = step_forward(x_1, weights[1], bias[1])
    #x_3 = step_forward(x_2, weights[2], bias[2])

def backward(x_3, weights, bias):
    pass

#def backward(x, weights, bias):
#    s_3 = final_layer(x[3])
#    s_2 = step_backward(s_3, x[2], weights[2], bias[2])
#    s_1 = step_backward(s_2, x[1], weights[1], bias[1])
#    s_0 = step_backward(s_1, x[0], weights[0], bias[0])
#    return [s_0, s_1, s_2, s_3]


def step_forward(prev_x, weights, bias):
    return helper.tanh(prev_x, weights, bias)


def step_backward(next_s, weights, bias):
    pass
