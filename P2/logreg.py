from collections import Set

import numpy as np

import mldata
import mlutil
import model


class LogisticRegression(model.Model):

	def __init__(self):
		# Add any additional model parameters
		super(LogisticRegression, self).__init__()
		self.weights = None

	def preprocess(self, data: mldata.ExampleSet):
		ndata = mlutil.convert_to_numpy(data)
		return mlutil.quantify_nominals(ndata[0], ndata[1])

	# randomly initialize weights for each feature
	# TODO Would it be possible to pass in truths when instantiating the
	#  class to keep the train() method signature the same as the other Model
	#  classes?
	def train(self, data: mldata.ExampleSet, truths: np.array):
		npdata = self.preprocess(data)
		weights = np.random.rand(len(npdata))
		# randomly initialize biases? not sure if i should even incorporate
		# biases
		bias = np.random.rand(len(npdata), 1)
		self.weights = self.gradient_descent(
			npdata, truths, weights, stepsize=1, skip=set())
		# okay now the weights should be finalized
		return self.weights, bias

	@staticmethod
	def sigmoid(x, bias=0):
		return 1 / (1 + np.exp(-x + bias))

	@staticmethod
	def weighted_input(x, weights):
		return np.dot(x, weights)

	def prob(self, weight, x):
		return self.sigmoid(self.weighted_input(weight, x))

	@staticmethod
	def conditional(x, y):
		# calculate log likelihood to maintain convexity instead of squared
		# loss
		return -y * np.math.log(x) - (1 - y) * np.math.log(1 - x)

	def predict(self, data: mldata.ExampleSet, weights, truths):
		# guesses = np.zeros(len(ndata[1]), 1) # use sigmoid to find guesses
		# guesses[np.where(sigmoid >= 0.5)] = 1 # truth guess when >= 0.5
		ndata = mlutil.convert_to_numpy(data)
		ndata = mlutil.quantify_nominals(ndata)
		weighted_feats = np.transpose(
			np.array([weights[i] * f for i, f in enumerate(ndata)]))
		sigmoid = np.array(
			# vector to save the sigmoid values for each example
			[self.sigmoid(sum(w)) for w in weighted_feats])
		out = self.conditional_log_likelihood(
			truths, sigmoid, weighted_feats, complexity=1) / len(sigmoid)
		# TODO loss is not used
		loss = out / len(sigmoid)  # average loss
		pass

	def conditional_log_likelihood(
			self,
			labels,
			sigmoids,
			weights,
			complexity=1):
		poslabel_idx = np.argwhere(labels > 0)
		neglabel_idx = np.argwhere(labels <= 0)
		pos_sigmoids = sigmoids[poslabel_idx]
		neg_sigmoids = sigmoids[neglabel_idx]
		w_squared = np.square(np.linalg.norm(weights))
		pos_exs = sum(-np.log(pos_sigmoids))
		neg_exs = sum(-np.log(1 - neg_sigmoids))
		neg_cond_log_like = pos_exs + neg_exs
		conditional_ll = (1 / 2) * w_squared + complexity * neg_cond_log_like
		return conditional_ll

	# identify the log loss and update parameters accordingly
	# call self recursively until no improvement (or not  enough improvement)
	# todo: for any binary variables, convert 0 and 1 to -1 and 1. look out
	#  for normalization
	# todo: check out overfitting control w/ c term and ||w||
	def gradient_descent(
			self,
			ndata,
			truths,
			weights,
			stepsize,
			epsilon=1,
			skip: Set = None):
		if skip is None:
			skip = set()
		iterations = 0

		while not (len(skip) == len(ndata) or iterations > 100):
			iterations = iterations + 1
			weighted_feats = []
			# for i, f in enumerate(ndata):
			# 	weighted_feats.append(weights[i] * f)
			weighted_feats = np.transpose(
				# theta^T*x: multiply all examples for each feature value by
				# its corresponding weight
				np.array([weights[i] * f for i, f in enumerate(ndata)]))
			# gradient calculated as dJ(theta)/d(theta_j) = 1/m (sum from 1 to
			# m of (h(x^i) - y^i) * x^ij
			# TODO Just make this a function?
			gradient_func = lambda x_sum, x, y: (self.sigmoid(x_sum) - y) * x
			# the calculations themselves give us ONE gradient for one
			# feature.
			# so we loop over all features and store results as a numpy array
			gradient = np.zeros(len(ndata))
			for j in range(len(ndata)):
				if j not in skip:
					# i indexes examples
					inv = 1 / len(ndata)
					grad = np.sum(
						gradient_func(
							sum(weighted_feats[:][i]), ndata[j][i], truths[i])
						for i, f in enumerate(ndata[j]))
					gradient[j] = inv * grad

			# todo: check which is local minimum. when loss stops decreasing?
			#  or is it when the derivative changes sign? important distinction
			finished = np.argwhere(abs(gradient) < epsilon)
			if len(finished) > 0:
				[skip.add(f[0]) for f in finished]
			weights = weights - stepsize * gradient
		# weights = weights[0]
		# self.gradient_descent(
		# 	ndata, truths, weights + stepsize * gradient, stepsize)
		return weights
