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

# X: m x n
# W: m x c
# z: c x n
# y: n x c
# m = 784
# c = 10
def softmaxRegression (trainingNumbers, trainingLabels, testingNumbers, testingLabels):
	n = np.shape(trainingNumbers)[1]
	nn = 100
	w = np.random.normal(0.0, 0.01, (784, 10))
	trainingNumbers = trainingNumbers.T
	randomState = np.random.get_state()
	np.random.shuffle(trainingNumbers)
	trainingNumbers = trainingNumbers.T
	np.random.set_state(randomState)
	np.random.shuffle(trainingLabels)
	epochs = 100
	for i in range(epochs):
		for j in range(n / nn):
			z = np.dot(w.T, trainingNumbers[:, j*nn:(j+1)*nn])
			yhat = (np.exp(z) / np.sum(np.exp(z), 0)).T
			gradient = (1./nn)*np.dot(trainingNumbers[:, j*nn:(j+1)*nn], (yhat - trainingLabels[j*nn:(j+1)*nn]))
			w = w - (0.016 * gradient)
		print i+1
		if ((epochs - i) <= 20):
			z = np.dot(w.T, trainingNumbers)
			yhatTrain = (np.exp(z) / np.sum(np.exp(z), 0)).T
			print "training percent correct:", fPC(trainingLabels, yhatTrain), "%"
	print
	zTest = np.dot(w.T, testingNumbers)
	yhatTest = (np.exp(zTest) / np.sum(np.exp(zTest), 0)).T
	print "testing percent correct:", fPC(testingLabels, yhatTest), "%"

def loadData (which):
	faces = np.load("mnist_{}_images.npy".format(which)).T
	labels = np.load("mnist_{}_labels.npy".format(which))
	return faces, labels

	
if __name__ == "__main__":
	print
	print "--------------------------------------------------------------"
	print "MACHINE LEARNING ALGORITHM TO READ HANDWRITTEN ARABIC NUMERALS"
	print "--------------------------------------------------------------"
	print
	testingNumbers, testingLabels = loadData("test")
	trainingNumbers, trainingLabels = loadData("train")
	softmaxRegression(trainingNumbers, trainingLabels, testingNumbers, testingLabels)
