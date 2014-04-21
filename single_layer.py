import numpy as np


def run_epoches(images, labels, n=200, eta=0.1):
	"""Run n number of epoches."""
	mse_weights = np.random.rand(784, 10)
	mse_bias = np.random.rand(10, 1)
	entropy_weights = np.random.rand(784, 10)
	entropy_bias = np.random.rand(10, 1)
	for i in range(n):
		batches = generate_batches(images, labels)
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
			print 'epoch=', i

	return (mse_weights, mse_bias, entropy_weights, entropy_bias)


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
		y = sigmoid(x, weights, bias)
		if b_or_w == 'w':
			v = np.dot(np.dot((y - t), (1 - y).T), y)
			addition = np.zeros((784, 10))
			col = 0
			for v_k in v:
				addition[:,col] = (v_k[0] * x).reshape(784)
				col += 1
			ret += addition
		else: # b_or_w == 'b':
			ret += np.dot(np.dot((y - t), (1 - y).T), y)
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
		y = sigmoid(x, weights, bias)
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


def sigmoid(x, w, b):
	"""Sigmoid function."""
	z = np.dot(w.T, x) + b
	return 1 / (1 + np.exp(-z))


def predict(images, weights, bias):
	"""Predict labels for images using trained weights and bias values."""
	prediction = []
	for im in images:
		y = sigmoid(im, weights, bias)
		prediction.append(np.argmax(y))
	return prediction


def calculate_error(predicted, true):
	"""Calculates error."""
	incorrect = 0.0
	index = 0
	while index < len(predicted):
		if predicted[index] != true[index]:
			incorrect += 1
	return incorrect / len(predicted)
