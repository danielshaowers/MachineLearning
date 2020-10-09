import functools
import math
from random import random
from typing import Any, Callable, Collection, Mapping, NoReturn, Optional, \
	Sequence, Tuple, Union

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
		laplace_smoothing_m: True will apply Laplace smoothing to the
			probabilities of the model parameters.
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
			laplace_smoothing_m: Union[int, float] = 0):
		super(NaiveBayes, self).__init__()
		if n_bins < 2:
			raise ValueError(
				'Number of bins must be at least 2 to discretize continuous '
				'features'
			)
		self.n_bins = n_bins
		self.laplace_smoothing_m = laplace_smoothing_m
		self.binners = dict()
		self.params = dict()

	def __repr__(self):
		class_name = f'{self.__class__.__name__}'
		n_bins = f'n_bins={self.n_bins}'
		laplace_smoothing = f'laplace_smoothing_m={self.laplace_smoothing_m}'
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
			A tuple of confidence values, each corresponding to a test example.

		Raises:
			AttributeError: If the model has not yet been trained.
			ValueError: If no examples are contained in the data.
		"""
		if len(data) == 0:
			raise ValueError('There are no examples to predict!')
		if len(self.params) == 0:
			raise AttributeError('Train the model first!')
		info = mlutil.get_features_info(data)
		preds = []
		label_info = mlutil.get_label_info(data)
		for ex in mlutil.get_example_features(data):
			probabilities = (
				# Compute the log prior-likelihood sum for each class label.
				[c, pr + sum(self._param(f, c, e) for e, f in zip(ex, info))]
				for c, pr in self.params[label_info].items()
			)
			# Get the class label (0) with the maximum probability (1).
			pred = max(probabilities, key=lambda pr: pr[1])
			# Log probability needs to be mapped back to get confidence.
			# TODO Normalize to proper probability
			conf = math.exp(pred[1])
			# TODO Return tuple of confidences
			preds.append(conf)
		return tuple(preds)

	def train(self, data: mldata.ExampleSet) -> NoReturn:
		"""Train the naive Bayes classifier.

		Prior to training, perform any necessary pre-processing of the data,
		such as discretizing continuous features.

		Args:
			data: Collection of training examples.

		Returns:
			None.
		"""
		self._preprocess(data)
		self._naive_bayes(data)

	def _naive_bayes(self, data: mldata.ExampleSet) -> NoReturn:
		model_parameters = dict()
		labels = mlutil.get_labels(data)
		label_info = mlutil.get_label_info(data)
		model_parameters[label_info] = metrics.probability(labels)
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		for feature, examples in feature_examples.items():
			exs = self._get_feature_values(feature, examples)
			model_parameters[feature] = self._compute_probability(exs, labels)
		self.params = model_parameters

	def _get_feature_values(
			self,
			feature: mldata.Feature,
			examples: Union[Any, Collection]) -> Union[Any, Collection]:
		"""Discretize the examples if continuous."""
		if len(self.binners) == 0:
			raise AttributeError('Preprocess the data first!')
		exs = examples
		if feature.type is mldata.Feature.Type.CONTINUOUS:
			if feature not in self.binners:
				raise KeyError('Feature binner function not found.')
			exs = self.binners[feature](examples)
		return exs

	def _compute_probability(
			self,
			event: Collection,
			given: Collection) -> Union[float, Mapping[Any, float]]:
		m = self.laplace_smoothing_m
		if self.laplace_smoothing_m < 0:
			m = len(set(event))
		p = 1 / m
		probability = metrics.probability(
			event=event, given=given, m=m, p=p, log_base=math.e)
		return probability

	def _preprocess(self, data: mldata.ExampleSet) -> NoReturn:
		"""Prefer space complexity over time complexity in terms of
		discretizing the continuous features. Each binner function will be
		used during training and prediction to determine the discretized
		value of the feature."""
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		self.binners = {
			f: self._train_binner(ex) for f, ex in feature_examples.items()
			if f.type == mldata.Feature.Type.CONTINUOUS
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
		f_value = self._get_feature_values(feature, feature_value)
		# Indexing is slightly different if looking up the label probability
		if feature_value is None:
			param = self.params[feature][label]
		else:
			param = self.params[feature][(f_value, label)]
		return param


def main(path: str, skip_cv: bool, n_bins: int, m: Union[int, float]):
	nb = NaiveBayes(n_bins=n_bins, laplace_smoothing_m=m)
	mainutil.p2_main(path, nb, skip_cv)


if __name__ == '__main__':
	random.seed(a=12345)
	parser = mainutil.base_arg_parser()
	parser.add_argument(
		'--m',
		type=float,
		help='Value for m-estimates. m < 0 will use Laplace smoothing')
	parser.add_argument(
		'--bins',
		type=int,
		help='Number of bins for continuous feature discretization (min 2)',
		required=True
	)
	args = parser.parse_args()
	main(path=args.path, skip_cv=args.skip_cv, n_bins=args.bins, m=args.m)
