import functools
import json
import math
from typing import Any, Callable, Collection, DefaultDict, Mapping, NoReturn, \
	Optional, Sequence, Tuple, Union

import jsonpickle
import numpy as np

import mainutil
import metrics
import mldata
import mlutil
import model


class NaiveBayes(model.Model):
	"""A standard naive Bayes classifier.

	Note that continuous features are discretized.

	Attributes:
		n_bins: Number of bins for discretizing continuous features. Must be
			at least 2 in order to discretize the features.
		laplace_m: True will apply Laplace smoothing to the probabilities of
			the model parameters.
		binners: A dictionary of partial discretization functions. Each
			function corresponds to a continuous feature to be binned. Prior
			to training the model, each continuous feature in the dataset is
			used to "train" a binner function to be used for both training
			and predicting.
		params: Learned model parameters from the training data.
	"""

	def __init__(
			self,
			n_bins: int = 2,
			laplace_m: Union[int, float] = 0,
			binners: Mapping[mldata.Feature, Callable[[Any], int]] = None,
			params: Mapping[mldata.Feature, Any] = None):
		super(NaiveBayes, self).__init__()
		if n_bins < 2:
			raise ValueError(
				'Number of bins must be at least 2 to discretize continuous '
				'features'
			)
		self.n_bins = n_bins
		self.laplace_m = laplace_m
		self.binners = dict() if binners is None else binners
		self.params = dict() if params is None else params

	def __repr__(self):
		class_name = f'{self.__class__.__name__}'
		n_bins = f'n_bins={self.n_bins}'
		laplace_smoothing = f'laplace_m={self.laplace_m}'
		return f'{class_name}({n_bins}, {laplace_smoothing})'

	def predict(self, data: mldata.ExampleSet) -> Tuple[float]:
		"""Predict the label of each test example in the data.

		For the naive Bayes classifier, the prior and likelihood are summed (
		in the case of log probabilities) or multiplied (in the case of
		standard probabilities) over all classes and examples. The class
		resulting in the maximum probability is selected for prediction.

		Args:
			data: Collection of test examples.

		Returns:
			A tuple of confidence values, each corresponding to a test
			example. Note that confidence values assume a binary class label
			and are relative to the "positive" class.

		Raises:
			AttributeError: If the model has not yet been trained.
			ValueError: If no examples are contained in the data.
		"""
		if len(data) == 0:
			raise ValueError('There are no examples to predict!')
		if len(self.params) == 0:
			raise AttributeError('Train the model first!')
		info = mlutil.get_features_info(data)
		label = mlutil.get_label_info(data)
		confidences = []
		for ex in mlutil.get_example_features(data):
			probabilities = [
				# Compute the log prior-likelihood sum for each class label.
				[c, pr + sum(self._param(f, c, e) for f, e in zip(info, ex))]
				for c, pr in self.params[label].items()
			]
			# Get the class label (0) with the maximum probability (1).
			prediction = max(probabilities, key=lambda pr: pr[1])
			# Normalization factor in computing the posterior probability.
			evidence = sum(math.exp(pr) for _, pr in probabilities)
			# Log probability needs to be mapped back to get confidence.
			confidence = math.exp(prediction[1]) / evidence
			# Keep all confidences relative to positive examples
			confidence = 1 - confidence if not prediction[0] else confidence
			confidences.append(confidence)
		return tuple(confidences)

	def train(self, data: mldata.ExampleSet) -> NoReturn:
		"""Train the naive Bayes classifier.

		Prior to training, perform any necessary pre-processing of the data,
		such as discretizing continuous features.

		Args:
			data: Collection of training examples.

		Returns:
			None. Model parameters are stored in the params attribute.
		"""
		self.binners = self._preprocess(data)
		self.params = self._naive_bayes(data)

	def _naive_bayes(
			self,
			data: mldata.ExampleSet) -> Mapping[mldata.Feature, Any]:
		model_parameters = dict()
		labels = mlutil.get_labels(data)
		label = mlutil.get_label_info(data)
		model_parameters[label] = self._compute_probability(labels)
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		for feature, examples in feature_examples.items():
			exs = self._get_feature_values(feature, examples)
			model_parameters[feature] = self._compute_probability(exs, labels)
		return model_parameters

	def _get_feature_values(
			self,
			feature: mldata.Feature,
			examples: Union[Any, Collection]) -> Union[Any, Collection]:
		"""Discretize the examples if continuous."""
		exs = examples
		if feature.type is mldata.Feature.Type.CONTINUOUS:
			if len(self.binners) == 0:
				raise AttributeError('Preprocess the data first!')
			if feature not in self.binners:
				raise KeyError('Feature binner function not found.')
			# Use the binner function to discretize the continuous feature.
			exs = self.binners[feature](examples)
		return exs

	def _compute_probability(
			self,
			event: Collection,
			given: Collection = None) -> DefaultDict[Any, float]:
		"""The default dictionary of probabilities accounts for missing
			feature value-class label combinations and uses the values of
			m and v from Laplace smoothing to determine the default value."""
		v = len(set(event))
		p = 1 / v
		m = v if self.laplace_m < 0 else self.laplace_m
		return metrics.probability(
			event=event, given=given, m=m, p=p, log_base=math.e)

	def _preprocess(
			self,
			data: mldata.ExampleSet) -> Mapping[mldata.Feature, Callable]:
		"""Prefer time complexity over space complexity in terms of
		discretizing the continuous features. Each binner function will be
		used during training and prediction to determine the discretized
		value of the feature."""
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		return {
			f: self._train_binner(ex) for f, ex in feature_examples.items()
			if f.type is mldata.Feature.Type.CONTINUOUS
		}

	def _train_binner(
			self,
			values: Sequence[Union[int, float]]) -> Callable:
		"""The partial function retains the split values, so an evaluated
		value will be the discretized value."""
		mx = max(values)
		mn = min(values)
		splits = np.arange(mn, mx, (mx - mn) / self.n_bins)
		return functools.partial(lambda to_bin: np.digitize(to_bin, splits))

	def _param(
			self,
			feature: mldata.Feature,
			label: Any,
			feature_value: Optional[Any] = None) -> Union[float, int]:
		"""Get the model parameter for the given feature."""
		# Indexing is slightly different if looking up the label probability
		if feature_value is None:
			param = self.params[feature][label]
		else:
			f_value = self._get_feature_values(feature, feature_value)
			param = self.params[feature][(f_value, label)]
		return param

	def save(self, file: str) -> NoReturn:
		with open(file, 'w') as f:
			saved = {
				'n_bins': self.n_bins,
				'laplace_m': self.laplace_m,
				'binners': {
					jsonpickle.dumps(feature): jsonpickle.dumps(binner)
					for feature, binner in self.binners.items()
				},
				'params': {
					jsonpickle.dumps(feature): param
					for feature, param in self.params.items()
				}
			}
			json.dump(saved, f)

	@staticmethod
	def load(file: str):
		with open(file) as f:
			learner = json.load(f)
			n_bins = learner['n_bins']
			laplace_m = learner['laplace_m']
			binners = {
				jsonpickle.loads(feature): jsonpickle.loads(binner)
				for feature, binner in learner['binners']
			}
			params = {
				jsonpickle.loads(feature): param
				for feature, param in learner['params']
			}
		return NaiveBayes(
			n_bins=n_bins,
			laplace_m=laplace_m,
			binners=binners,
			params=params
		)


def main(path: str, skip_cv: bool, n_bins: int, m: Union[int, float]):
	nb = NaiveBayes(n_bins=n_bins, laplace_m=m)
	mainutil.p2_main(path=path, learner=nb, skip_cv=skip_cv)


if __name__ == '__main__':
	import time

	start = time.time()
	main(path='..\\voting', skip_cv=False, n_bins=2, m=0)
	stop = time.time()
	print(f'RUNTIME: {round((stop - start) / 60, 3)} minutes')
# random.seed(a=12345)
# parser = mainutil.base_arg_parser()
# parser.add_argument(
# 	'--m',
# 	type=float,
# 	help='Value for m-estimates. m < 0 will use Laplace smoothing',
# 	required=True)
# parser.add_argument(
# 	'--bins',
# 	type=int,
# 	help='Number of bins for continuous feature discretization (min 2)',
# 	required=True
# )
# args = parser.parse_args()
# main(path=args.path, skip_cv=args.skip_cv, n_bins=args.bins, m=args.m)
