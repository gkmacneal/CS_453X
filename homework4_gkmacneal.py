from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm

class SVM453X ():
	def __init__ (self):
		pass

	# Expects each *row* to be an m-dimensional row vector. X should
	# contain n rows, where n is the number of examples.
	# y should correspondingly be an n-vector of labels (-1 or +1).

	def fit (self, X, y):
		m = np.shape(X)[1]
		n = np.shape(X)[0]
		Xp = np.append(X.T, [np.ones(n)], axis=0).T
		G = np.matrix(y).dot(Xp) * -1.
		P = Xp.T.dot(Xp) # m+1 x m+1
		q = np.zeros(m+1) # 1 x m+1
		h = -1.

		# Solve -- if the variables above are defined correctly, you can call this as-is:
		sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

		# Fetch the learned hyperplane and bias parameters out of sol['x'] m+1 x 1
		self.w = sol['x'][:m]
		self.b = sol['x'][m]

	# Given a 2-D matrix of examples X, output a vector of predicted class labels
	def predict (self, x):
		yhat = x.dot(self.w) + self.b
		print "yhat:"
		print yhat
		yhatPos = np.greater_equal(yhat, 0).astype(int)
		yhatNeg = np.less(yhat, 0).astype(int) * -1
		yhatF = (yhatPos + yhatNeg).T
		# print yhatF
		return yhatF

def test1 ():
	# Set up toy problem
	X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
	y = np.array([-1,-1,-1,1,1,1])

	# Train your model
	svm453X = SVM453X()
	svm453X.fit(X, y)
	print "w:"
	print svm453X.w
	print "b:"
	print svm453X.b

	# Compare with sklearn
	svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
	svm.fit(X, y)
	print(svm.coef_, svm.intercept_)

	acc = np.mean(svm453X.predict(X) == svm.predict(X))
	print("Acc={}".format(acc))

def test2 ():
	# Generate random data
	X = np.random.rand(20,3)
	# Generate random labels based on a random "ground-truth" hyperplane
	while True:
		w = np.random.rand(3)
		y = 2*(X.dot(w) > 0.5) - 1
		# Keep generating ground-truth hyperplanes until we find one
		# that results in 2 classes
		if len(np.unique(y)) > 1:
			break

	print y
	svm453X = SVM453X()
	svm453X.fit(X, y)
	print "w:"
	print svm453X.w
	print "b:"
	print svm453X.b

	# Compare with sklearn
	svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
	svm.fit(X, y)
	diff = np.linalg.norm(svm.coef_ - svm453X.w) + np.abs(svm.intercept_ - svm453X.b)
	print "diff:"
	print(diff)

	acc = np.mean(svm453X.predict(X) == svm.predict(X))
	print("Acc={}".format(acc))
	if acc == 1 and diff < 1e-1:
		print("Passed")

if __name__ == "__main__": 
	test1()
	test2()
