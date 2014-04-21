import scipy.io
import numpy as np
from sklearn.preprocessing import normalize
from single_layer import run_epoches, predict, calculate_error


if __name__ == '__main__':
	train = scipy.io.loadmat('data/train.mat')
	train_images = train['train']['images'][0][0] # (28, 28, 60000) ndarray
	train_labels = train['train']['labels'][0][0] # (60000, 1) ndarray
	# (60000, 784) ndarray
	train_images = \
		np.reshape(np.transpose(train_images, [2, 0, 1]), (60000, 784))
	train_images = normalize(train_images.astype(float))

	mse_weights, mse_bias, entropy_weights, entropy_bias = \
		run_epoches(train_images, train_labels)

	test = scipy.io.loadmat('data/test.mat')
	test_images = test['test']['images'][0][0]
	test_labels = test['test']['labels'][0][0]
	test_images = \
		np.reshape(np.transpose(test_images, [2, 0, 1]), (10000, 784))
	test_images = normalize(test_images.astype(float))
	test_labels = [x[0] for x in test_labels] # convert to list

	# return predicted labels as lists
	predicted_mse_labels = predict(test_images, mse_weights, mse_bias)
	predicted_entropy_labels = predict(test_images, entropy_weights, entropy_bias)

	mse_error = calculate_error(predicted_mse_labels, test_labels)
	entropy_error = calculate_error(predicted_entropy_labels, test_labels)
	print mse_error
	print entropy_error
	