import functools
import random
from typing import Set

import numpy as np

import mainutil
import mldata
import mlutil
import model


class LogisticRegression(model.Model):

	def __init__(self, cost: float = 1):
		super(LogisticRegression, self).__init__()
		self.weights = None
		self.cost = cost

	@staticmethod
	def preprocess(data: mldata.ExampleSet):
		np_data, f_types = mlutil.convert_to_numpy(data)
		return mlutil.quantify_nominals(np_data, f_types)

	def train(self, data: mldata.ExampleSet):
		truths = mlutil.get_labels(data)
		np_data = self.preprocess(data)
		# randomly initialize weights for each feature
		weights = np.random.rand(len(np_data))
		# randomly initialize biases? should i even incorporate biases
		bias = np.random.rand(len(np_data), 1)
		self.weights = self.gradient_descent(
			np_data, truths, weights, stepsize=1, skip=set()
		)

	# okay now the weights should be finalized

	@staticmethod
	@functools.lru_cache(512)
	def sigmoid(x, bias=0):
		return 1 / (1 + np.exp(-x + bias))

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
		return log_likelihood_scores >= 0.5, log_likelihood_scores

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

	# identify the log loss and update parameters accordingly
	# call self recursively until no improvement (or not  enough improvement)
	# todo: for any binary variables, convert 0 and 1 to -1 and 1. look out
	#  for normalization
	# todo: check out overfitting control w/ c term and ||w||
	@functools.lru_cache(512)
	def gradient_descent(self, ndata, truths, weights, stepsize, epsilon=1,
						 skip: Set = None):
		if skip is None:
			skip = set()
		iterations = 0
		while not (len(skip) == len(ndata) or iterations > 1000):
			iterations = iterations + 1
			weighted_feats = []
			#	for i, f in enumerate(ndata):
			#		weighted_feats.append(weights[i] * f)
			weighted_feats = np.transpose(
				np.array([weights[i] * f for i, f in enumerate(ndata)])
			)  #
			# theta^T*x: multiply all examples for each feature value by its
			# corresponding weight
			# gradient calculated as dJ(theta)/d(theta_j) = 1/m (sum from 1 to
			# m of (h(x^i) - y^i) * x^ij
			gradient_func = lambda x_sum, x, y: (self.sigmoid(x_sum) - y) * x
			# the calculations themselves give us ONE gradient for one
			# feature. so we loop over all features and store results as a
			# numpy array
			gradient = np.zeros(len(ndata))
			for j in (jj for jj in range(len(ndata)) if jj not in skip):
				summation = sum(
					gradient_func(
						sum(weighted_feats[:][i]), ndata[j][i], truths[i])
					for i, f in enumerate(ndata[j])  # i indexes examples
				)
				gradient[j] = (1 / len(ndata)) * summation
			# todo: check which is local minimum.
			finished = np.argwhere(abs(gradient) < epsilon)
			# when loss stops decreasing? or is it when the derivative changes
			# sign? important distinction
			if len(finished) > 0:
				[skip.add(f[0]) for f in finished]
			weights = weights - stepsize * gradient
		# weights = weights[0]
		# self.gradient_descent(ndata, truths, weights + stepsize *
		# gradient, stepsize)
		return weights


def main(path: str, skip_cv: bool, cost: float):
	learner = LogisticRegression(cost=cost)
	mainutil.p2_main(path, learner, skip_cv)


if __name__ == "__main__":
	random.seed(a=12345)
	parser = mainutil.base_arg_parser()
	parser.add_argument(
		'--cost',
		type=float,
		help='Cost term for negative conditional log likelihood'
	)
	args = parser.parse_args()
	main(path=args.path, skip_cv=args.skip_cv, cost=args.cost)
