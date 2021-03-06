#!/usr/bin/env python

import numpy as np
np.set_printoptions(threshold=np.nan)
import math

def fPC (y, yhat):
	# find the correct guesses
	correct = np.equal(np.argmax(y, 1), np.argmax(yhat, 1))
	# find the mean of correct guesses
	PC = np.mean(correct)*100.
	return PC

def fCE(y, yhat):
	labels = np.argmax(y, 1)
	m = np.shape(labels)[0]
	return np.sum(-1.*np.log(yhat[range(m), labels])) / m

def softmax(z):
	return (np.exp(z) / np.sum(np.exp(z), 0)).T

def relu(z):
	return z * np.greater(z, 0).astype(int)

def reluPrime(z):
	return np.greater(z, 0).astype(float)

def neuralNetwork (trainingNumbers, trainingLabels, validNumbers, validLabels, hSize, rate, nn, epochs, alpha1, alpha2, beta1, beta2):
	print "hidden layer size:", hSize
	print "rate:", rate
	print "batch size:", nn
	print "epochs:", epochs
	print "regularization strengths:", alpha1, alpha2, beta1, beta2
	print
	n = np.shape(trainingNumbers)[1]
	m = np.shape(trainingNumbers)[0]
	c = np.shape(trainingLabels)[1]
	# randomize initial weights
	W1 = np.random.normal(0.0, 1./math.sqrt(hSize), (m, hSize))
	W2 = np.random.normal(0.0, 1./math.sqrt(c), (hSize, c))
	b1 = np.random.normal(0.0, 0.1, (hSize,))
	b2 = np.random.normal(0.0, 0.1, (c,))
	# shuffle the training set
	trainingNumbers = trainingNumbers.T
	randomState = np.random.get_state()
	np.random.shuffle(trainingNumbers)
	trainingNumbers = trainingNumbers.T
	np.random.set_state(randomState)
	np.random.shuffle(trainingLabels)
	print "training..."
	for i in range(epochs):
		for j in range(n / nn):
			# z1 should take it from 784(m) by nn to hSize by nn
			z1 = (np.dot(W1.T, trainingNumbers[:, j*nn:(j+1)*nn]).T + b1).T
			h1 = relu(z1)
			# z2 should take it from hSize by nn to 10(c) by nn
			z2 = (np.dot(W2.T, h1).T + b2).T
			yhat = softmax(z2)
			# gradient calculations
			diff = yhat - trainingLabels[j*nn:(j+1)*nn]
			g = diff.dot(W2.T) * reluPrime(z1.T)
			gb1 = np.sum(g, 0)
			gW1 = (trainingNumbers[:, j*nn:(j+1)*nn]).dot(g) + (alpha1 * W1) + (beta1 * np.sign(W1))
			gb2 = np.sum(diff)
			gW2 = h1.dot(diff) + (alpha2 * W2) + (beta2 * np.sign(W2))
			b1 = b1 - (rate * gb1)
			W1 = W1 - (rate * gW1)
			b2 = b2 - (rate * gb2)
			W2 = W2 - (rate * gW2)
		print i+1
		printStuff = True
		if ((epochs - i) <= 20 and printStuff):
			z1 = (np.dot(W1.T, trainingNumbers).T + b1).T
			h1 = relu(z1)
			z2 = (np.dot(W2.T, h1).T + b2).T
			yhatTrain = softmax(z2)
			tCEloss = fCE(trainingLabels, yhatTrain)
			tPC = fPC(trainingLabels, yhatTrain)
			print "training cross entropy loss:", tCEloss
			print "training percent correct:", tPC, "%"
			print
	print
	# calculate accuracy of final weights on training data
	z1 = (np.dot(W1.T, trainingNumbers).T + b1).T
	h1 = relu(z1)
	z2 = (np.dot(W2.T, h1).T + b2).T
	yhatTrain = softmax(z2)
	tCEloss = fCE(trainingLabels, yhatTrain)
	tPC = fPC(trainingLabels, yhatTrain)
	print "training cross entropy loss:", tCEloss
	print "training percent correct:", tPC, "%"
	print
	# calculate accuracy of final weights on validation data
	z1 = (np.dot(W1.T, validNumbers).T + b1).T
	h1 = relu(z1)
	z2 = (np.dot(W2.T, h1).T + b2).T
	yhatValid = softmax(z2)
	vCEloss = fCE(validLabels, yhatValid)
	vPC = fPC(validLabels, yhatValid)
	print "validation cross entropy loss:", vCEloss
	print "validation percent correct:", vPC, "%"
	print
	# pass weights and accuracy out to the hyperparameter optimization
	return vPC, W1, b1, W2, b2

def findBestHyperparameters(trainingNumbers, trainingLabels, validNumbers, validLabels):
	# list hyperparameters to try here
	hSizes = [30, 30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 60, 60, 60, 60, 60]
	rates = [0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001]
	batches = [64, 64, 256, 256, 64, 64, 256, 256, 64, 64, 256, 256, 64, 64, 256, 256]
	epochs = 50
	alpha1s = [0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75]
	alpha2s = [0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75, 0.1, 0.75]
	beta1s = [0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01]
	beta2s = [0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01]
	bestPC = 0.
	bestW = (0, 0, 0, 0, 0)
	for i in range(len(hSizes)):
		(PC, W1, b1, W2, b2) = neuralNetwork(trainingNumbers, trainingLabels, validNumbers, validLabels, hSizes[i], rates[i], batches[i], epochs, alpha1s[i], alpha2s[i], beta1s[i], beta2s[i])
		if PC > bestPC:
			bestPC = PC
			bestW = (i, W1, b1, W2, b2)
	print "Of those tested, best hyperparameters found:"
	print "hidden layer size:", hSizes[bestW[0]]
	print "rate:", rates[bestW[0]]
	print "batch size:", batches[bestW[0]]
	print "epochs:", epochs
	print "regularization strengths:", alpha1s[bestW[0]], alpha2s[bestW[0]], beta1s[bestW[0]], beta2s[bestW[0]]
	return bestW[1], bestW[2], bestW[3], bestW[4]

def loadData (which):
	numbers = np.load("mnist_{}_images.npy".format(which)).T
	labels = np.load("mnist_{}_labels.npy".format(which))
	return numbers, labels


if __name__ == "__main__":
	print
	print "-------------------------------------------------------------"
	print " NEURAL NETWORK TRAINING TO READ HANDWRITTEN ARABIC NUMERALS "
	print "-------------------------------------------------------------"
	print
	testingNumbers, testingLabels = loadData("test")
	trainingNumbers, trainingLabels = loadData("train")
	validNumbers, validLabels = loadData("validation")
	# train with various hyperparameters
	W1, b1, W2, b2 = findBestHyperparameters(trainingNumbers, trainingLabels, validNumbers, validLabels)
	# calculate the accuracy of the final hyper parameters
	z1 = (np.dot(W1.T, testingNumbers).T + b1).T
	h1 = relu(z1)
	z2 = (np.dot(W2.T, h1).T + b2).T
	yhatTest = softmax(z2)
	testCEloss = fCE(testingLabels, yhatTest)
	testPC = fPC(testingLabels, yhatTest)
	print "testing cross entropy loss:", testCEloss
	print "testing percent correct:", testPC, "%"
