from typing import Sequence, Tuple

import numpy as np

import metrics
import mldata
import mlutil
import model


class NaiveBayes(model.Model):

	def __init__(self, k_bins: int = 2, use_laplace_smoothing: bool = False):
		super(NaiveBayes, self).__init__()
		if k_bins < 2:
			raise ValueError("""k_bins must be at least 2 to discretize 
			continuous features""")
		self.k_bins = k_bins
		self.use_laplace_smoothing = use_laplace_smoothing
		self.quantiles = None
		self.params = None

	def predict(self, data: mldata.ExampleSet):
		pass

	def train(self, data: mldata.ExampleSet):
		feature_array, label_array = self.preprocess(data)
		self._naive_bayes(feature_array, label_array)

	def _naive_bayes(self, data: mldata.ExampleSet):
		labels = mlutil.get_labels(data)
		label_tests = mlutil.create_discrete_split_tests(labels)
		label_probs = metrics.probability(labels)
		"""
		For each feature (all categorical after "quantilization")...
			compute probability for each combination of label and feature
			record as parameter of the naive bayes model
		"""

	def preprocess(self, data: mldata.ExampleSet):
		# Prevent pre-processing again; self.quantiles is "final"
		if self.quantiles is not None:
			return
		else:
			self.quantiles = dict()
		features_info = mlutil.get_features_info(data, as_tuple=False)
		continuous_features = (
			(i, f) for i, f in enumerate(features_info)
			if f.type == mldata.Feature.Type.CONTINUOUS)
		feature_array, label_array = self.get_feature_label_split(data)
		for i, f in continuous_features:
			# Feature array does not include ID feature
			f_values = feature_array[:, i + 1]
			# Quantiles are analogous to partitions
			n_quantiles = self.k_bins - 1
			quantiles, quantilized = self.quantilize(f_values, n_quantiles)
			# Store for lookup later when predicting
			self.quantiles[f] = quantiles
			feature_array[:, i] = np.array(quantilized)
		return feature_array, label_array

	@staticmethod
	def get_feature_label_split(data: mldata.ExampleSet):
		data_array = np.array(data.to_float())[1:]
		feature_array = data_array[:, :-1]
		label_array = data_array[::, -1]
		return feature_array, label_array

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
