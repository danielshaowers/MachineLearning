import functools
import json
import random
from typing import NoReturn, Set

import numpy as np
from scipy.stats import stats

import mainutil
import mldata
import mlutil
import model
import preprocess


class LogisticRegression(model.Model):

	def __init__(self, cost: float = 0.1, iterations=10, weights=None, fold=1, stepsize=0.5):
		super(LogisticRegression, self).__init__()
		self.weights = weights
		self.cost = cost
		self.iterations = 500
		self.fold = fold
		self.stepsize = stepsize
		self.types =None

	def __repr__(self):
		class_name = f'{self.__class__.__name__}'
		cost = f'cost={self.cost}'
		iterations = f'iterations={self.iterations}'
		return f'{class_name}({cost}, {iterations})'

	def preprocess(self, data: mldata.ExampleSet):
		np_data, f_types = mlutil.convert_to_numpy(data)
		self.types = f_types
		np_data = mlutil.quantify_nominals(np_data, f_types)
		np_data =  np.asarray(np_data, dtype='float64')
		np_data = preprocess.normalize(np_data, f_types)
		np_data = preprocess.adjust_binary(np_data, f_types)
		return np_data
	def train(self, data: mldata.ExampleSet):
		truths = np.array(mlutil.get_labels(data))
		# perform preprocessing of data: quantify nominal features
		np_data = self.preprocess(data)
		# randomly initialize weights for each feature
		weights = np.random.rand(len(np_data))
		self.weights = self.gradient_descent(
			np_data, truths, weights, stepsize=self.stepsize, skip=set()
		)
		self.save(self.get_name() + str(self.fold) + "f_" + str(self.iterations) + "it_" + str(self.cost) + "cost_" + "length" + str(len(np_data)) + "step" + str(self.stepsize))
	# okay now the weights should be finalized

	@staticmethod
	def sigmoid(x, bias=0):
		z = np.exp(-x - bias)
		return 1 / (1 + z)


	@staticmethod
	def conditional(x, y):
		# calculate log likelihood to maintain convexity instead of squared loss
		return -y * np.math.log(x) - (1 - y) * np.math.log(1 - x)

	def predict(self, data: mldata.ExampleSet):
		np_data = self.preprocess(data)
		weighted_feats = np.transpose(
			np.array([self.weights[i] * f for i, f in enumerate(np_data)]))
		# vector to save the sigmoid values for each example
		log_likelihood_scores = np.array([
			self.sigmoid(sum(w)) for w in weighted_feats
		])
		return log_likelihood_scores


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
		return  (self.sigmoid(x_sum) - y) * x

	# identify the log loss and update parameters accordingly
	def gradient_descent(self, ndata, truths, weights, stepsize, epsilon=0.001,
						 skip: Set = None):
		if skip is None:
			skip = set()
		iterations = 0
		while not (len(skip) == len(ndata) or iterations > self.iterations):
			iterations = iterations + 1
			counter = 0
			weighted_feats = np.transpose(
				np.array([np.multiply(weights[i],f) for i, f in enumerate(ndata)])
			)
			gradient = np.zeros(len(ndata))
			#run sigmoid function on wx -
			sig = self.sigmoid(np.sum(weighted_feats, axis=1), bias= 0) - truths #-(np.sum(weights) / 2))
			res = np.zeros(len(ndata) - len(skip))
			for j in (jj for jj in range(len(ndata)) if jj not in skip):
				res[counter] = np.sum(sig * ndata[j])
				# the derivative of conditional log likelihood with R
				gradient[j] = (1 / len(ndata[0])) * (res[counter]+ self.cost * weights[j])
				counter = counter + 1
			finished = np.argwhere(abs(gradient) < epsilon) # find which gradients we can stop
			if len(finished) > 1:
				[skip.add(f[0]) for f in finished]

			weights = weights - stepsize * gradient
		return weights

	def get_name(self):
		return self.__class__.__name__

	def save(self, file: str) -> NoReturn:
		with open(file, 'w') as f:
			saved = {
				'weights': self.weights.tolist(),
				'cost': self.cost,
				'iterations': self.iterations,
				'fold': self.fold
			}
			json.dump(saved, f, indent='\t')

	@staticmethod
	def load(file: str):
		with open(file) as f:
			learner = json.load(f)
			weights = learner['weights']
			cost = learner['cost']
			iters = learner['iterations']
			fold = learner['fold']
		return LogisticRegression(cost=cost, iterations=iters, weights=weights, fold=fold)


def main(path: str, skip_cv: bool, cost: float, iterations=500):
	learner = LogisticRegression(cost=cost, iterations=iterations, stepsize=0.5)
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
	main(path='..\\voting', skip_cv=True, cost=0.1, iterations=1000)
