import argparse

def make_parser():

	parser = argparse.ArgumentParser()
	parser.add_argument('-train_file', type=str, required=True,
		help='Path to the training file (CSV only)')
	parser.add_argument('-validation_file', type=str, required=True,
		help='Path to the validation file (CSV only)')
	parser.add_argument('-method', type=str, choices=['naive_bayes', 'perceptron', 'logistic_reg'], default='logistic_reg',
		help='choose one method for training: naive_bayes (Naive Bayes), perceptron, logistic_reg (logistic regression)')

	return parser

