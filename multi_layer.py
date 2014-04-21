import numpy as np
import helper
import matplotlib.pyplot as plt


def run_epoches(images, labels, n=200, eta=0.1):
	"""Run n number of epoches."""
	# initialize weights and bias values to random
	mse_weights = [np.random.rand(784, 300), np.random.rand(784, 100), \
		np.random.rand(784, 10)]
	mse_bias = [np.random.rand(300, 1), np.random.rand(100, 1), \
		np.random.rand(10, 1)]
	entropy_weights = [np.random.rand(784, 300), np.random.rand(784, 100), \
		np.random.rand(784, 10)]
	entropy_bias = [np.random.rand(300, 1), np.random.rand(100, 1), \
		np.random.rand(10, 1)]
	for i in range(n):
		batches = generate_batches(images, labels)
		for batch in batches:		
			x_mse_w = forward(images, mse_weights, mse_bias)
			sigma_mse_w = backward(x_mse_w, mse_weights, mse_bias)
			x_entropy_w = forward(images, entropy_weights, entropy_bias)
			sigma_entropy_w = backward(x_entropy_w, entropy_weights, entropy_bias)

			x_mse_b = forward(images, mse_weights, mse_bias)
			sigma_mse_b = backward(x_mse_b, mse_weights, mse_bias)
			x_entropy_b = forward(images ,entropy_weights, entropy_bias)
			sigma_entropy_b = backward(x_entropy_b, entropy_weights, entropy_bias)
			
			mse_weights = update(x_mse_w, sigma_mse_w, eta)
			mse_bias = update(x_mse_b, sigma_mse_b, eta)
			entropy_weights = update(x_entropy_w, sigma_entropy_w, eta)
			entropy_bias = update(x_entropy_b, sigma_entropy_b, eta)
	return (mse_weights, mse_bias, entropy_weights, entropy_bias)


def forward(x_0, weights, bias):
	x_1 = step_forward(x_0, weights[0], bias[0])
	x_2 = step_forward(x_1, weights[1], bias[1])
	x_3 = step_forward(x_2, weights[2], bias[2])
	return [x_0, x_1, x_2, x_3]


def backward(x, weights, bias):
	s_3 = final_layer(x[3])
	s_2 = step_backward(s_3, x[2], weights[2], bias[2])
	s_1 = step_backward(s_2, x[1], weights[1], bias[1])
	s_0 = step_backward(s_1, x[0], weights[0], bias[0])
	return [s_0, s_1, s_2, s_3]


def step_forward(prev_x, weights, bias):
	return tanh(prev_x, weights, bias)


def step_backward(next_s, weights, bias):
	pass