import functools
import math
from typing import Callable, Collection, Sequence, Tuple

import numpy as np

import metrics
import mldata
import mlutil
import model


class NaiveBayes(model.Model):

	def __init__(self, n_bins: int = 2, use_laplace_smoothing: bool = False):
		super(NaiveBayes, self).__init__()
		if n_bins < 2:
			raise ValueError("""Number of bins must be at least 2 to 
			discretize continuous features""")
		self.n_bins = n_bins
		self.use_laplace_smoothing = use_laplace_smoothing
		self.binners = dict()
		self.params = dict()

	def predict(self, data: mldata.ExampleSet):
		f_info = mlutil.get_features_info(data)
		if any((f not in self.params for f in f_info)):
			raise ValueError("""Data to predict must include all of the same 
			features that were used to train the model.""")
		label_info = mlutil.get_label_info(data)
		labels = mlutil.get_labels(data)
		examples = mlutil.get_example_features(data)
		preds = []
		for ex, label in zip(examples, labels):
			label_probs = [
				# For each class label, sum all probabilities that correspond
				# to the values of the example
				# TODO Double check (e, c) -- do we want Pr(event|given)?
				# TODO Still need to add Pr(C) for class label
				(c, sum(self.params[f][(e, c)] for e, f in zip(ex, f_info)))
				for c in self.params[label_info]
			]
			# Get the class label (0) with the maximum probability (1)
			pred = max(label_probs, key=lambda x: x[1])
			preds.append(model.Prediction(value=pred[0], confidence=pred[1]))
		return preds

	def train(self, data: mldata.ExampleSet):
		self.preprocess(data)
		self._naive_bayes(data)

	def _naive_bayes(self, data: mldata.ExampleSet):
		model_parameters = dict()
		labels = mlutil.get_labels(data)
		label_info = mlutil.get_label_info(data)
		model_parameters[label_info] = metrics.probability(labels)
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		for feature, examples in feature_examples.items():
			exs = examples
			if feature.type is feature.Type.CONTINUOUS:
				binner = self.binners[feature]
				exs = binner(examples)
			model_parameters[feature] = self._compute_probability(exs, labels)
		self.params = model_parameters

	def _compute_probability(self, event: Collection, given: Collection):
		if self.use_laplace_smoothing:
			m = len(set(event))
			p = 1 / m
			probability = metrics.probability(
				event=event, given=given, m=m, p=p, log_base=math.e)
		else:
			probability = metrics.probability(
				event=event, given=given, log_base=math.e)
		return probability

	def preprocess(self, data: mldata.ExampleSet):
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		self.binners = {
			f: self.train_binner(ex) for f, ex in feature_examples.items()
			if f.type == mldata.Feature.Type.CONTINUOUS
		}

	def train_binner(self, values: Sequence) -> Callable:
		mx = max(values)
		mn = min(values)
		splits = np.arange(mn, mx, (mx - mn) / self.n_bins)
		return functools.partial(lambda to_bin: np.digitize(to_bin, splits))

	# TODO Maybe use for research extension?
	@staticmethod
	def quantilize(values: Sequence, n: int = 2) -> Tuple:
		quantiles = NaiveBayes.compute_quantiles(values, n)
		discretized = []
		for v in values:
			i_quantile = 0
			while v > quantiles[i_quantile]:
				i_quantile += 1
			discretized.append(i_quantile)
		return quantiles, tuple(discretized)

	@staticmethod
	def compute_quantiles(values: Sequence, n: int = 2):
		if n < 2:
			raise ValueError('n must be at least 2')
		return tuple(
			np.quantile(values, q=(quant + 1) / n) for quant in range(n - 1))


if __name__ == '__main__':
	data = mldata.parse_c45('spam', '..')
	nb = NaiveBayes()
	nb.train(data)
