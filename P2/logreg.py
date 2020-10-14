import functools

import json
import random
from typing import Set

from typing import NoReturn, Set


import jsonpickle
import numpy as np

import model, mainutil

import mldata
import mlutil
import preprocess


class LogisticRegression(model.Model):

	def __init__(self, cost: float = 0.1, iterations=100, fold =1):
		super(LogisticRegression, self).__init__()
		self.weights = None
		self.cost = cost
		self.iterations = iterations
		self.fold = fold

	@staticmethod
	def preprocess(data: mldata.ExampleSet):
		data = preprocess.standardize(data)
		np_data, f_types = mlutil.convert_to_numpy(data)
		np_data = mlutil.quantify_nominals(np_data, f_types)
		return np.asarray(np_data, dtype='float64')

	def train(self, data: mldata.ExampleSet):
		truths = np.array(mlutil.get_labels(data))
		np_data = self.preprocess(data)
		# randomly initialize weights for each feature
		weights = np.random.rand(len(np_data))
		# randomly initialize biases? should i even incorporate biases
		bias = np.random.rand(len(np_data), 1)
		self.weights = self.gradient_descent(
			np_data, truths, weights, stepsize=1, skip=set()
		)
		self.save(self.getName() + str(self.fold) + "f_" + str(self.iterations) + "it_" + str(self.cost) + "cost_" + "length" + str(len(np_data)))
	# okay now the weights should be finalized

	@staticmethod
	@functools.lru_cache(512)
	#numerically stable sigmoid to prevent overflow errors
	def sigmoid(x, bias=0):
		if x >= 0:
			z = np.exp(-x)
			return 1 / (1 + z)
		else:
			# if x is less than zero then z will be small, denom can't be
			# zero because it's 1+z.
			z = np.exp(x)
			return z / (1 + z)

	@staticmethod
	@functools.lru_cache(512)
	def weighted_input(x, weights):
		return np.dot(x, weights)

	@functools.lru_cache(512)
	def prob(self, weight, x):
		return self.sigmoid(self.weighted_input(weight, x))

	@staticmethod
	@functools.lru_cache(512)
	def conditional(x, y):
		# calculate log likelihood to maintain convexity instead of squared
		# loss
		return -y * np.math.log(x) - (1 - y) * np.math.log(1 - x)

	@functools.lru_cache(512)
	def predict(self, data: mldata.ExampleSet):
		# guesses = np.zeros(len(ndata[1]), 1) # use sigmoid to find guesses
		# guesses[np.where(sigmoid >= 0.5)] = 1 # truth guess when >= 0.5
		np_data = self.preprocess(data)
		weighted_feats = np.transpose(
			np.array([self.weights[i] * f for i, f in enumerate(np_data)]))
		# vector to save the sigmoid values for each example
		log_likelihood_scores = np.array([
			self.sigmoid(sum(w)) for w in weighted_feats
		])

		return log_likelihood_scores

	# return [model.Prediction(value=sc > 0.5, confidence=sc) for i,
	# sc in enumerate(log_likelihood_scores)]
	@functools.lru_cache(512)
	def conditional_log_likelihood(self, labels, sigmoids, weights):
		poslabel_idx = np.argwhere(labels > 0)
		neglabel_idx = np.argwhere(labels <= 0)
		pos_sigmoids = sigmoids[poslabel_idx]
		neg_sigmoids = sigmoids[neglabel_idx]
		squared_norm = np.square(np.linalg.norm(weights))
		pos_log_sigmoids = sum(-np.log(pos_sigmoids))
		neg_log_sigmoids = sum(-np.log(1 - neg_sigmoids))
		neg_log_cond_like = pos_log_sigmoids + neg_log_sigmoids
		conditional_ll = 0.5 * squared_norm + self.cost * neg_log_cond_like
		return conditional_ll
	#derivative of conditional log likelihood 1/2 ||w||^2 is 2x
	def gradient_func(self, weights, weight, x_sum, x, y):
		#return self.cost * weight / np.math.sqrt(sum(np.square(weights))) + (self.sigmoid(x_sum) - y) * x
		return  (self.sigmoid(x_sum) - y) * x

	# identify the log loss and update parameters accordingly
	# repeat
	# todo: for any binary variables, convert 0 and 1 to -1 and 1. look out
	#  for normalization
	# todo: check out overfitting control w/ c term and ||w||
	def gradient_descent(self, ndata, truths, weights, stepsize, epsilon=1,
						 skip: Set = None):
		if skip is None:
			skip = set()
		iterations = 0
		while not (len(skip) == len(ndata) or iterations > self.iterations):
			iterations = iterations + 1
			weighted_feats = np.transpose(
				np.array([np.multiply(weights[i],f) for i, f in enumerate(ndata)])
			)  #
			# theta^T*x: multiply all examples for each feature value by its
			# corresponding weight
			# the calculations themselves give us ONE gradient for one
			# feature. so we loop over all features and store results as an array
			gradient = np.zeros(len(ndata))
			for j in (jj for jj in range(len(ndata)) if jj not in skip):
				summation = sum(
					self.gradient_func(
						weights, weights[j], sum(weighted_feats[:][i]), ndata[j][i], truths[i])
					for i, f in enumerate(ndata[j])  # i indexes examples
				)
				gradient[j] = (1 / len(ndata)) * summation
			finished = np.argwhere(abs(gradient) < epsilon)
			if len(finished) > 0:
				[skip.add(f[0]) for f in finished]
			# the derivative of ||W||^2 wrt any parameter is just that parameter.
			weights = weights - stepsize * (self.cost * (sum(weights)) + gradient)  #(1/len(weights)) * self.cost * weight- (stepsize * self.cost / len(ndata)) * weights # update weights
		return weights


	def getName(self):
		return "logreg"

	def save(self, file: str):
		with open(file, 'w') as f:
			saved = {
				'weights': self.weights.tolist(),
				'cost': self.cost,
				'iterations': self.iterations,
				'fold': self.fold
			}
			json.dump(saved, f)

	@staticmethod
	def load(file: str):
		with open(file) as f:
			learner = json.load(f)
			weights = learner['weights']
			cost = learner['cost']
			iterations = learner['iterations']
			fold = learner['fold']
		return (
			learner, weights, cost, iterations, fold
		)


def main(path: str, skip_cv: bool, cost: float, iterations=100):
	learner = LogisticRegression(cost=cost, iterations = iterations,)
	mainutil.p2_main(path, learner, skip_cv)


if __name__ == "__main__":

	random.seed(a=12345)
	main(path = '..\\volcanoes', skip_cv = 0, cost=0.1)
#	parser = mainutil.base_arg_parser()
#	parser.add_argument(
#		'--cost',
#		type=float,
#		help='Cost term for negative conditional log likelihood'
#	)
#	args = parser.parse_args()
#	main(path=args.path, skip_cv=args.skip_cv, cost=args.cost)
