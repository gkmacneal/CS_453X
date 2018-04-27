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
	return np.greater(z, 0).astype(int)

def neuralNetwork (trainingNumbers, trainingLabels, validNumbers, validLabels, hSize, rate, nn, epochs, alpha1, alpha2, beta1, beta2):
	print "hidden layer size:", hSize
	print "rate:", rate
	print "batch size:", nn
	print "epochs:", epochs
	print "regularization strengths:", alpha1, alpha2, beta1, beta2
	n = np.shape(trainingNumbers)[1]
	m = np.shape(trainingNumbers)[0]
	c = np.shape(trainingLabels)[1]
	print "n:", n
	print "m:", m
	print "c:", c
	W1 = np.random.normal(0.0, 1./math.sqrt(hSize), (m, hSize))
	print W1
	W2 = np.random.normal(0.0, 1./math.sqrt(c), (hSize, c))
	b1 = np.random.normal(0.0, 0.1, (hSize,))
	b2 = np.random.normal(0.0, 0.1, (c,))
	trainingNumbers = trainingNumbers.T
	randomState = np.random.get_state()
	np.random.shuffle(trainingNumbers)
	trainingNumbers = trainingNumbers.T
	np.random.set_state(randomState)
	np.random.shuffle(trainingLabels)
	for i in range(epochs):
		for j in range(n / nn):
			# z1 should take it from 784(m) by nn to hSize by nn
			z1 = (np.dot(W1.T, trainingNumbers[:, j*nn:(j+1)*nn]).T + b1).T
			h1 = relu(z1)
			# z2 should take it from hSize by nn to 10(c) by nn
			z2 = np.dot(W2.T, h1).T + b2
			yhat = softmax(z2)
			diff = yhat.T - trainingLabels[j*nn:(j+1)*nn]
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
		print W1
		"""
		print
		z1 = (np.dot(W1.T, trainingNumbers).T + b1).T
		h1 = relu(z1)
		z2 = (np.dot(W2.T, h1).T + b2).T
		yhatTrain = softmax(z2)
		tCEloss = fCE(trainingLabels, yhatTrain)
		tPC = fPC(trainingLabels, yhatTrain)
		print "training cross entropy loss:", tCEloss
		print "training percent correct:", tPC, "%"
		"""
	print
	z1 = (np.dot(W1.T, trainingNumbers).T + b1).T
	h1 = relu(z1)
	z2 = (np.dot(W2.T, h1).T + b2).T
	yhatTrain = softmax(z2)
	tCEloss = fCE(trainingLabels, yhatTrain)
	tPC = fPC(trainingLabels, yhatTrain)
	print "training cross entropy loss:", tCEloss
	print "training percent correct:", tPC, "%"
	print
	z1 = (np.dot(W1.T, validNumbers).T + b1).T
	h1 = relu(z1)
	z2 = (np.dot(W2.T, h1).T + b2).T
	yhatValid = softmax(z2)
	vCEloss = fCE(validLabels, yhatValid)
	vPC = fPC(validLabels, yhatValid)
	print "validation cross entropy loss:", vCEloss
	print "validation percent correct:", vPC, "%"
	return vCEloss

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
	neuralNetwork(trainingNumbers, trainingLabels, validNumbers, validLabels, 50, 0.0001, 64, 50, 50, 50, 50, 50)
