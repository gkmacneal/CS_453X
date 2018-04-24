#!/usr/bin/env python

import numpy as np
#np.set_printoptions(threshold=np.nan)
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
	# find the correct guesses
	correct = np.equal(np.argmax(y, 1), np.argmax(yhat, 1))
	# find the mean of correct guesses
	PC = np.mean(correct)*100.
	return PC

def fCE(y, yhat):
	labels = np.argmax(y, 1)
	m = np.shape(labels)[0]
	return np.sum(-1.*np.log(yhat[range(m), labels]) / m

# X: m x n
# W: m x c
# z: c x n
# y: n x c
def neuralNetwork (trainingNumbers, trainingLabels, validNumbers, validLabels, hSize, rate, nn, epochs, regularization):
	n = np.shape(trainingNumbers)[1]
	m = np.shape(trainingNumbers)[0]
	c = np.shape(trainingLabels)[1]
	W1 = np.random.normal(0.0, 1./sqrt(c), (m, c))
	W2 = np.random.normal(0.0, 1./sqrt(c), (hSize, c))
	trainingNumbers = trainingNumbers.T
	randomState = np.random.get_state()
	np.random.shuffle(trainingNumbers)
	trainingNumbers = trainingNumbers.T
	np.random.set_state(randomState)
	np.random.shuffle(trainingLabels)
	for i in range(epochs):
		for j in range(n / nn):
			z = np.dot(w.T, trainingNumbers[:, j*nn:(j+1)*nn])
			yhat = (np.exp(z) / np.sum(np.exp(z), 0)).T
			gradient = (1./nn)*np.dot(trainingNumbers[:, j*nn:(j+1)*nn], (yhat - trainingLabels[j*nn:(j+1)*nn]))
			w = w - (rate * gradient)
		print i+1
		if ((epochs - i) <= 20):
			z = np.dot(w.T, trainingNumbers)
			yhatTrain = (np.exp(z) / np.sum(np.exp(z), 0)).T
			print "training cross entropy loss:", fCE(trainingLabels, yhatTrain)
	print
	zValid = np.dot(w.T, validNumbers)
	yhatValid = (np.exp(zValid) / np.sum(np.exp(zValid), 0)).T
	print "validation cross entropy loss:", fCE(validLabels, yhatValid)

def loadData (which):
	numbers = np.load("mnist_{}_images.npy".format(which)).T
	labels = np.load("mnist_{}_labels.npy".format(which))
	return numbers, labels

	
if __name__ == "__main__":
	print
	print "--------------------------------------------------------------"
	print "MACHINE LEARNING ALGORITHM TO READ HANDWRITTEN ARABIC NUMERALS"
	print "--------------------------------------------------------------"
	print
	testingNumbers, testingLabels = loadData("test")
	trainingNumbers, trainingLabels = loadData("train")
	validNumbers, validLabels = loadData("vailidation")
	neuralNetwork(trainingNumbers, trainingLabels, validNumbers, validLabels, 50, 0.01, 64, 50, )





