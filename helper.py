import numpy as np


def sigmoid(x, w, b):
    """Sigmoid function."""
    z = np.dot(w.T, x) + b
    ret = 1.0 / (1 + np.exp(-z))
    ret[ret == 1] = .99
    ret[ret == 0] = .01
    return ret


def tanh(x, w, b):
    z = np.dot(w.T, x) + b
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def generate_batches(images, labels):
    """Randomly shuffles data and divides into batches of 200."""
    appended = np.append(images, labels, axis=1)
    np.random.shuffle(appended)
    batches = []
    start_i = 0
    for i in range(200, len(labels) + 1, 200):
        batches.append(appended[start_i:i])
        start_i = i
    return batches


def error(weights, bias, images, labels):
    """Calculates the classification accuracy using weights and bias values."""
    predicted_labels = predict(images, weights, bias)
    error = calculate_error(predicted_labels, labels)
    return error


def predict(images, weights, bias):
    """Predict labels for images using weights and bias values."""
    prediction = []
    for im in images:
        y = sigmoid(im.reshape(784, 1), weights, bias)
        prediction.append(np.argmax(y))
    return prediction


def calculate_error(predicted, true):
    """Calculates error between predicted and true."""
    true = [x[0] for x in true] # convert to list
    incorrect = 0.0
    index = 0
    while index < len(predicted):
        if predicted[index] != true[index]:
            incorrect += 1
        index += 1
    return incorrect / len(predicted)
