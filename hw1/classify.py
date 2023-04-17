#!/bin/python
from sklearn import svm
from sklearn import neural_network as nn
def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(C = 3)
	# cls = svm.SVC(kernel='linear', C = 0.08)
	# cls = svm.LinearSVC(penalty = 'l2', C = 0.65, dual= True)
	# cls = nn.MLPClassifier(hidden_layer_sizes = (200,) ,early_stopping = True)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy", acc)

