import functools
from typing import Callable, Sequence, Tuple

import numpy as np

import metrics
import mldata
import mlutil
import model


class NaiveBayes(model.Model):

	def __init__(self, n_bins: int = 2, use_laplace_smoothing: bool = False):
		super(NaiveBayes, self).__init__()
		if n_bins < 2:
			raise ValueError("""k_bins must be at least 2 to discretize 
			continuous features""")
		self.n_bins = n_bins
		self.use_laplace_smoothing = use_laplace_smoothing
		self.binners = dict()
		self.params = None

	def predict(self, data: mldata.ExampleSet):
		pass

	def train(self, data: mldata.ExampleSet):
		self.preprocess(data)
		self._naive_bayes(data)

	def _naive_bayes(self, data: mldata.ExampleSet):
		params = dict()
		labels = mlutil.get_labels(data)
		label_info = mlutil.get_label_info(data)
		params[label_info] = metrics.probability(labels)
		feature_examples = mlutil.get_feature_examples(data, as_dict=True)
		for feature, examples in feature_examples.items():
			exs = examples
			if feature.type is feature.Type.CONTINUOUS:
				binner = self.binners[feature]
				exs = binner(examples)
			# TODO Maybe clean this up -- namedtuple or class?
			# 	To get parameter: self.params[feature][(value, label)]
			if self.use_laplace_smoothing:
				m = len(set(exs))
				p = 1 / m
				pr = metrics.probability(event=exs, given=labels, m=m, p=p)
			else:
				pr = metrics.probability(event=exs, given=labels)
			params[feature] = pr
		self.params = params

	def preprocess(self, data: mldata.ExampleSet):
		f_exs = mlutil.get_feature_examples(data, as_dict=True)
		self.binners = {
			f: self.train_binner(ex) for f, ex in f_exs.items()
			if f.type == mldata.Feature.Type.CONTINUOUS
		}

	# TODO Wait until done to see if we can use this
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
