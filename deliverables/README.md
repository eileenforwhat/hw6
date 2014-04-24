Partners: Eileen Li, Minyoon Jung
cs189-el, cs189-hk
22824915, 22453592

Dependencies:
scikit-learn
python

How to run code:
In main.py: 

	sl.run_epoches(train_images, train_labels, test_images, test_labels, epoches=100, alpha=0.6)

	sl.run_epoches(train_images, train_labels, test_images, test_labels, epoches=100, alpha=0.6)

These two functions will run the given number of epoches, report error rates for all four methods (mse + bias, mse + weights, entropy + bias, entropy + weights) every 10 epoches. It will also generate accuracy graphs.
