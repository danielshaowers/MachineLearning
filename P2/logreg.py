import json
import random
from typing import NoReturn, Set

import numpy as np

import mainutil
import mldata
import mlutil
import model
import preprocess


class LogisticRegression(model.Model):

	def __init__(
			self, cost: float = 0.1,
			iterations: int = 1000,
			step_size: float = 0.5,
			weights=None):
		super(LogisticRegression, self).__init__()
		self.weights = weights
		self.cost = cost
		self.iterations = iterations
		self.step_size = step_size
		self.types = None

	def __repr__(self):
		class_name = f'{self.__class__.__name__}'
		cost = f'cost={self.cost}'
		iterations = f'iterations={self.iterations}'
		return f'{class_name}({cost}, {iterations})'

	def preprocess(self, data: mldata.ExampleSet):
		# data = preprocess.standardize(data)
		np_data, f_types = mlutil.convert_to_numpy(data)
		self.types = f_types
		np_data = mlutil.quantify_nominals(np_data, f_types)
		np_data = np.asarray(np_data, dtype='float64')
		np_data = preprocess.normalize(np_data, f_types)
		np_data = preprocess.adjust_binary(np_data, f_types)
		return np_data

	def train(self, data: mldata.ExampleSet):
		truths = np.array(mlutil.get_labels(data))
		np_data = self.preprocess(data)
		# randomly initialize weights for each feature
		weights = np.random.rand(len(np_data))
		# randomly initialize biases? should i even incorporate biases
		bias = np.random.rand(len(np_data), 1)
		self.weights = self.gradient_descent(
			np_data, truths, weights, step_size=self.step_size, skip=set()
		)

	# okay now the weights should be finalized

	@staticmethod
	# numerically stable sigmoid to prevent overflow errors
	def sigmoid(x, bias=0):
		z = np.exp(-x - bias)
		return 1 / (1 + z)

	@staticmethod
	def conditional(x, y):
		# calculate log likelihood to maintain convexity instead of squared
		# loss
		return -y * np.math.log(x) - (1 - y) * np.math.log(1 - x)

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

	# derivative of conditional log likelihood 1/2 ||w||^2 is 2x
	def gradient_func(self, x_sum, x, y):
		return (self.sigmoid(x_sum) - y) * x

	# identify the log loss and update parameters accordingly
	# repeat
	# todo: for any binary variables, convert 0 and 1 to -1 and 1. look out
	#  for normalization
	# todo: check out overfitting control w/ c term and ||w||
	def gradient_descent(self, ndata, truths, weights, step_size, epsilon=0.001,
						 skip: Set = None):
		if skip is None:
			skip = set()
		iterations = 0
		while not (len(skip) == len(ndata) or iterations > self.iterations):
			iterations = iterations + 1
			counter = 0
			weighted_feats = np.transpose(
				np.array([np.multiply(weights[i], f) for i, f in enumerate(ndata)])
			)
			# weighted_feats = preprocess.normalize(weighted_feats)#
			# theta^T*x: multiply all examples for each feature value by its
			# corresponding weight
			# the calculations themselves give us ONE gradient for one
			# feature. so we loop over all features and store results as an
			# array
			gradient = np.zeros(len(ndata))
			sig = self.sigmoid(np.sum(weighted_feats, axis=1),bias=0) - truths  # -(np.sum(weights) / 2))
			res = np.zeros(len(ndata) - len(skip))
			for j in (jj for jj in range(len(ndata)) if jj not in skip):  #
				res[counter] = np.sum(sig * ndata[j])
				# calculate the gradient wrt each feature
				# summation = sum(
				# 	self.gradient_func(
				# 		weights, weights[j], sum(weighted_feats[:][i]),
				# 		ndata[j][i], truths[i])
				# 	for i, f in enumerate(ndata[j])  # i indexes examples
				# )
				gradient[j] = (1 / len(ndata[0])) * (
						res[counter] + self.cost * weights[j])
				counter = counter + 1
			# find which gradients we can stop
			finished = np.argwhere(abs(gradient) < epsilon)
			if len(finished) > 1:
				[skip.add(f[0]) for f in finished]
			# the derivative of ||W||^2 wrt any parameter is just that
			# parameter.
			# (1/len(weights)) * self.cost * weight- (stepsize * self.cost /
			# len(ndata)) * weights # update weights
			weights = weights - step_size * gradient  # (self.cost * (sum(
		# weights)) + gradient)
		return weights

	def get_name(self):
		return self.__class__.__name__

	def save(self, file: str) -> NoReturn:
		with open(file, 'w') as f:
			saved = {
				'weights': self.weights.tolist(),
				'cost': self.cost,
				'iterations': self.iterations
			}
			json.dump(saved, f, indent='\t')

	@staticmethod
	def load(file: str):
		with open(file) as f:
			learner = json.load(f)
			weights = learner['weights']
			cost = learner['cost']
			iters = learner['iterations']
		return LogisticRegression(cost=cost, iterations=iters, weights=weights)


def main(path: str, skip_cv: bool, cost: float, iterations=1000):
	learner = LogisticRegression(
		cost=cost, iterations=iterations, step_size=0.5
	)
	mainutil.p2_main(path, learner, skip_cv)


def command_line_main():
	random.seed(a=12345)
	parser = mainutil.base_arg_parser()
	parser.add_argument(
		'--cost',
		type=float,
		help='Cost term for negative conditional log likelihood'
	)
	args = parser.parse_args()
	main(path=args.path, skip_cv=args.skip_cv, cost=args.cost)


if __name__ == "__main__":
	# command_line_main()
	main(path='..\\volcanoes', skip_cv=False, cost=0.1)
