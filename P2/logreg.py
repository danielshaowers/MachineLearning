import mldata

import model
import numpy as np
import mlutil
import math
import random
class LogisticRegression(model.Model):


	def __init__(self):
		# Add any additional model parameters
		super(LogisticRegression, self).__init__()


	def train(self, data: mldata.ExampleSet, truths: np.array):
		ndata = mlutil.convert_to_numpy(data)
		weights = np.random.rand(len(ndata),1) # randomly initialize weights for each feature

		weights = self.gradient_descent(ndata, truths, weights, math.inf,  0.03)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def predict(self, data: mldata.ExampleSet):
		#guesses = np.zeros(len(ndata[1]), 1) # use sigmoid to find guesses
		#guesses[np.where(sigmoid >= 0.5)] = 1 # truth guess when >= 0.5
		pass
	# identify the log loss and update parameters accordingly
	#call self recursively until
	def gradient_descent(self, ndata, truths, weights, prev_loss, stepsize):
		weighted_feats = np.array([weights[i] * f for i, f in ndata])  # theta^T*x: multiply all examples for each feature value by its corresponding weight
		sigmoid = np.array([self.sigmoid(sum(w)) for w in weighted_feats]) # vector to save the sigmoid values for each feature

		loss_func = lambda x,y: -y * np.math.log(x) - (1 - y) * np.math.log(1 - x) # calculate log loss to maintain convexity instead of squared loss
		out = [loss_func(g, truths) for e,g in enumerate(sigmoid)]
		loss = sum(out) / len(out) # loss value
		if prev_loss < loss: #todo: check which is local minimum. when loss stops decreasing? or is it when the derivative changes sign? important distinction
			return weights
		else: # not at a minimum yet
			# find derivative of sigmoid
			# gradient calculated as dJ(theta)/d(theta_j) = 1/m (sum from 1 to m of (h(x^i) - y^i) * x^ij
			gradient_func = lambda x_sum, x, y: (sigmoid(x_sum) - y) * x
			# for j in range = for every feature
			# for i in enumerate(ndata[j]) = for the j'th feature of every example
			# the calculations themselves give us ONE gradient for one feature. so we loop over all features and store results as a numpy array
			gradient = np.array([(1/len(ndata)) * np.sum(gradient_func(weighted_feats[j], truths[i], ndata[i][j]) for i, f in enumerate(ndata[j])) for j in range(len(ndata))])
			self.gradient_descent(ndata, truths, weights + stepsize * gradient, loss, stepsize)
