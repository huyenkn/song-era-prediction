import os
import sys
import argparse
import time
import random
random.seed(1234)

import numpy as np
np.random.seed(1234)

import pandas as pd

import scipy as sp

import torch
from torch import optim

from gtnlplib import preproc
from gtnlplib import clf_base
from gtnlplib import constants
from gtnlplib import evaluation
from gtnlplib import naive_bayes
from gtnlplib import perceptron
from gtnlplib import logreg

import flags

def preprocessing(file):

	#Bags of word: y_tr: list of labels (4000); x_tr: list of counters (4000)
	y_list,x_list = preproc.read_data(file,preprocessor=preproc.bag_of_words)

	#Unseen words: counts: a corpus Counter (9006)
	counts = preproc.aggregate_counts(x_list)

	#Pruning the vocabulary
	#x_pruned: list of counters (4000) that contain only words that appear at least min_counts times in counts_tr
	#vocab: a set of words (4875)
	x_pruned, vocab = preproc.prune_vocabulary(counts,x_list,10)

	return y_list, counts, x_pruned, vocab

def naive_bayes_method():

	print('Training', args.method, '...')

	smoother_vals = np.logspace(-1,2,5)
	
	#best_smoother, scores = naive_bayes.find_best_smoother(
	#	x_tr_pruned, y_tr, x_dv_pruned, y_dv, smoother_vals)
	#theta_nb = naive_bayes.estimate_nb(x_tr_pruned, y_tr, best_smoother)
	best_smoother, theta_nb = naive_bayes.find_best_smoother(
		x_tr_pruned, y_tr, x_dv_pruned, y_dv, smoother_vals)
	#inference:
	y_hat = clf_base.predict_all(x_dv_pruned, theta_nb, labels)

	# this shows how we write and read predictions for evaluation
	#evaluation.write_predictions(y_hat,'nb-dev.preds')
	print('Evaluating', args.method, '...')
	evaluation.write_predictions(y_hat, args.validation_file + '.preds')
	
	#y_hat_dv = evaluation.read_predictions('nb-dev.preds')
	
	#return evaluation.acc(y_hat_dv,y_dv)
	return evaluation.acc(y_hat, y_dv)

def perception_method():

	print('Training', args.method, '...')

	theta_perc,theta_perc_history = perceptron.estimate_perceptron(x_tr_pruned,y_tr,20)

	#inference:
	# to write the predictions on the dev and training data
	y_hat_dv = clf_base.predict_all(x_dv_pruned,theta_perc,labels)

	print('Evaluating', args.method, '...')
	evaluation.write_predictions(y_hat_dv, args.validation_file + '.preds')
	#y_hat_te = clf_base.predict_all(x_te_pruned,theta_perc,labels)
	#evaluation.write_predictions(y_hat_te,'perc-test.preds')

	#y_hat = evaluation.read_predictions('perc-dev.preds')
	
	return evaluation.acc(y_hat_dv,y_dv)

def logistic_reg_method():

	print('Training', args.method, '...')
	X_tr = preproc.make_numpy(x_tr_pruned,vocab)
	X_dv = preproc.make_numpy(x_dv_pruned,vocab)

	Y_tr = np.array([label_set.index(y_i) for y_i in y_tr])
	Y_dv = np.array([label_set.index(y_i) for y_i in y_dv])

	X_tr_var = torch.from_numpy(X_tr.astype(np.float32))
	X_dv_var = torch.from_numpy(X_dv.astype(np.float32))

	# build a new model with a fixed seed
	torch.manual_seed(765)
	model = logreg.build_linear(X_tr,Y_tr)
	model.add_module('softmax',torch.nn.LogSoftmax(dim=1))

	loss = torch.nn.NLLLoss()

	Y_tr_var = torch.from_numpy(Y_tr)
	Y_dv_var = torch.from_numpy(Y_dv)

	#logP = model.forward(X_tr_var)
	#logreg.nll_loss(logP.data.numpy(), Y_tr)

	model_trained, losses, accuracies = logreg.train_model(loss,model,
                                                       X_tr_var,
                                                       Y_tr_var,
                                                       X_dv_var=X_dv_var,
                                                       Y_dv_var = Y_dv_var,
                                                       num_its=1000,
                                                       optim_args={'lr':0.009})
	
	#inference: after training model, use the trained model to make prediction on an evaluation set.
	_, Y_hat_dv = model_trained.forward(X_dv_var).max(dim=1)
	print('Evaluating', args.method, '...')

	Y_np = Y_hat_dv.data.numpy()
	Y_hat_dv_list = []
	for i in range(Y_np.shape[0]):
		Y_hat_dv_list.append(label_set[Y_np[i]])

	Y_hat_dv_np = np.array(Y_hat_dv_list)

	evaluation.write_predictions(Y_hat_dv_np, args.validation_file + '.preds')

	# np.save('logreg-es-dev.preds.npy', Y_hat_dv.data.numpy())
	
	return evaluation.acc(np.load('logreg-es-dev.preds.npy'),Y_dv_var.data.numpy())


if __name__ == "__main__":

	parser = flags.make_parser()
	args = parser.parse_args()

	if not os.path.exists(args.train_file):
		print('Train file %s not exists!' % args.train_file)
		sys.exit(1)

	if not os.path.exists(args.validation_file):
		print('Validation file %s not exists!' % args.validation_file)
		sys.exit(1)

	# if not os.path.exists(args.test_file):
	# 	print('Test file %s not exists!' % args.test_file)
	# 	sys.exit(1)

	y_tr, counts_tr, x_tr_pruned, vocab = preprocessing(args.train_file)
	y_dv, counts_dv, x_dv_pruned, _ = preprocessing(args.validation_file)
	labels = set(y_tr)
	label_set = sorted(list(labels))

	if args.method == 'naive_bayes':
		accuracy = naive_bayes_method()
	elif args.method == 'perceptron':
		accuracy = perception_method()
	else:
		accuracy = logistic_reg_method()

	print('Accuracy:', accuracy)
























