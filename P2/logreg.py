import json
import random
from typing import NoReturn

import numpy as np

import adaboost
import mainutil
import mldata
import P2.mlutil as mlutil
import model
import preprocess

# only change made was we changed the weights to reflect the weight of each example.
# so if an example has feature val of 0.7 and example weight is 0.2, then its weight becomes 0.14
class LogisticRegression(model.Model):

	def __init__(
			self,
			cost: float = 0.1,
			iterations: int = 1000,
			step_size: float = 0.5,
			weights=None,
			boost_weights = None):
		super(LogisticRegression, self).__init__()
		self.weights = weights
		self.cost = cost
		self.iterations = iterations
		self.step_size = step_size
		self.boost_weights = boost_weights

	def __repr__(self):
		class_name = f'{self.__class__.__name__}'
		cost = f'cost={self.cost}'
		iterations = f'iterations={self.iterations}'
		return f'{class_name}({cost}, {iterations})'


	def preprocess(self, data: mldata.ExampleSet):
		np_data, f_types = mlutil.convert_to_numpy(data)
		np_data = mlutil.quantify_nominals(np_data, f_types)
		np_data = np.asarray(np_data, dtype='float64')
		np_data = preprocess.normalize(np_data, f_types)
		np_data = preprocess.adjust_binary(np_data, f_types)
		return np_data

	def train(self, data: mldata.ExampleSet):
		if self.boost_weights == None:
			self.boost_weights = np.ones(len(data))
		truths = np.array(mlutil.get_labels(data))
		# perform pre-processing of data: quantify nominal features
		np_data = self.preprocess(data)
		self.weights = self.gradient_descent(np_data, truths)


	# okay now the weights should be finalized
	@staticmethod
	def sigmoid(x, bias=0):
		z = np.exp(-x - bias)
		return 1 / (1 + z)

	@staticmethod
	def conditional(x, y):
		# calculate log likelihood to maintain convexity instead of squared
		# loss
		return -y * np.log(x) - (1 - y) * np.log(1 - x)

	def predict(self, data: mldata.ExampleSet):
		if len(data) == 0:
			raise ValueError('There are no examples to predict!')
		if len(self.weights) == 0:
			raise AttributeError('Train the model first!')
		np_data = self.preprocess(data)
		weighted = [self.weights[i] * f for i, f in enumerate(np_data)]
		weighted = np.array(weighted).T
		# vector to save the sigmoid values for each example
		log_likelihoods = np.array([self.sigmoid(sum(w)) for w in weighted])
		return log_likelihoods

	def conditional_log_likelihood(self, labels, sigmoids, weights):
		poslabel_idx = np.argwhere(labels > 0)
		neglabel_idx = np.argwhere(labels <= 0)
		pos_sigmoids = sigmoids[poslabel_idx]
		neg_sigmoids = sigmoids[neglabel_idx]
		squared_norm = np.square(np.linalg.norm(weights))
		pos_log_sigmoids = adaboost.wsum(-np.log(pos_sigmoids), self.boost_weights[poslabel_idx])
		neg_log_sigmoids = adaboost.wsum(-np.log(1 - neg_sigmoids), self.boost_weights[neglabel_idx])
		neg_log_cond_like = pos_log_sigmoids + neg_log_sigmoids
		conditional_ll = 0.5 * squared_norm + self.cost * neg_log_cond_like
		return conditional_ll

	# derivative of conditional log likelihood 1/2 ||w||^2 is 2x
	@staticmethod
	def gradient_func(x_sum, x, y):
		return (LogisticRegression.sigmoid(x_sum) - y) * x

	# identify the log loss and update parameters accordingly
	# repeat
	def gradient_descent(self, data, truths, epsilon=0.001):
		# randomly initialize weights for each feature
		weights = np.random.rand(len(data))
		skip = set()
		iterations = 0
		while not (len(skip) == len(data) or iterations > self.iterations):
			iterations += 1
			weighted = [weights[i] * f for i, f in enumerate(data)]
			weighted = np.array(weighted).T

			# theta^T*x: multiply all examples for each feature value by its
			# corresponding weight
			# the calculations themselves give us ONE gradient for one
			# feature. so we loop over all features and store results as an
			# array
			gradient = np.zeros(len(data))
			sig = self.sigmoid(np.sum(weighted, axis=1), bias=0) - truths
			results = np.zeros(len(data) - len(skip))
			features_idx = [f for f in range(len(data)) if f not in skip]
			total_boosted_weight = sum(self.boost_weights)
			for i, f in enumerate(features_idx):
				results[i] = np.sum(sig * data[f] * self.boost_weights)
				# calculate the gradient wrt each feature
				gradient[f] = (1 / total_boosted_weight) * (
						results[i] + self.cost * weights[f])
			# find which feature gradients we can stop
			finished = np.argwhere(abs(gradient) < epsilon)
			skip.update(f[0] for f in finished)
			weights = weights - self.step_size * gradient
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
	mainutil.p2_main(path=path, learner=learner, skip_cv=skip_cv)


def command_line_main():
	random.seed(a=12345)
	parser = mainutil.base_arg_parser()
	parser.add_argument(
		'--cost',
		type=float,
		help='Cost term for negative conditional log likelihood',
		required=True
	)
	args = parser.parse_args()
	main(path=args.path, skip_cv=args.skip_cv, cost=args.cost)


if __name__ == "__main__":
	#command_line_main()
	 main(path='..\\voting', skip_cv=True, cost=0.1, iterations=1000)
