from typing import Generator, Iterable, Mapping, NoReturn, Sequence, Tuple

import logreg
import mldata
import model
import nbayes


class NaiveLogisticRegression(model.Model):
	"""A logistic regression model that either partially or fully depends on
	the prediction of a naive Bayes model when the confidence of a predicted
	class label by logistic regression is within some margin of 0.5.

	Under the hood, a naive logistic regression model is a fully trained naive
	Bayes model and a fully trained logistic regression model, assumed to be
	trained on the same training data, or at least on data from the same
	dataset.

	The margin is a real-valued number between 0 and 1, inclusive, that defines
	the region around 0.5 that is considered "uncertain." When making
	predictions, there are two possibilities. If the confidence of a
	prediction by logistic regression is outside the margin, then the
	prediction made my logistic regression is used. If the confidence falls
	within the margin, then naive logistic regression can be set to either
	defer fully to the prediction of the naive Bayes, or by a rank-based
	approach. In the latter case, the confidences from naive Bayes and
	logistic regression are respectively ranked. Let T be the total number of
	examples classified with a confidence above 0.5 from naive Bayes and
	logistic regression. Then, an example E is predicted as being from the
	positive class (label = 1) if

			naive_bayes_rank(E) + logistic_regression_rank(E) >= T

	That is, if the sum of the confidence rankings from both models exceeds
	the total number of examples predicted with a confidence above 0.5. The
	intuition behind this threshold is that only examples predicted with
	sufficient relative confidence by naive Bayes and logistic regression
	will be predicted as the positive label. By using rank, skew in the
	confidence distribution does not affect the decision metric. This
	primarily applies to naive Bayes in which confidences will be
	concentrated around 0 and 1.
	"""

	def __init__(
			self,
			naive_bayes: nbayes.NaiveBayes,
			log_reg: logreg.LogisticRegression,
			use_rank: bool = False,
			margin: float = 0.1,
			re_train: bool = False):
		super(NaiveLogisticRegression, self).__init__()
		self.naive_bayes = naive_bayes
		self.log_reg = log_reg
		self.use_rank = use_rank
		self._check_margin(margin)
		self.margin = margin
		self.re_train = re_train

	@staticmethod
	def _check_margin(margin: float):
		if margin < 0 or 1 < margin:
			raise ValueError('Margin must be between 0 and 1 inclusive')

	def train(self, data: mldata.ExampleSet) -> NoReturn:
		if self.re_train:
			self.naive_bayes.train(data)
			self.log_reg.train(data)

	def predict(self, data: mldata.ExampleSet) -> Tuple:
		nb_predictions = self.naive_bayes.predict(data)
		lr_predictions = self.log_reg.predict(data)
		uncertain = self._get_uncertain_idx(lr_predictions)
		if self.use_rank:
			predictions, n_adjustments = self._predict_by_rank(
				uncertain, lr_predictions, nb_predictions, lr_predictions
			)
		else:
			predictions, n_adjustments = self._predict_by_naive_bayes(
				uncertain, lr_predictions, nb_predictions
			)
		return predictions, n_adjustments

	def _get_uncertain_idx(self, x: Sequence[float]) -> Generator:
		return (
			i for i in range(len(x))
			if 0.5 - self.margin <= x[i] <= 0.5 + self.margin
		)

	@staticmethod
	def _predict_by_rank(
			uncertain_idx: Iterable[int],
			initial_predictions: Iterable[float],
			nb_predictions: Sequence[float],
			lr_predictions: Sequence[float]) -> Tuple[Tuple[int], int]:
		ranked_nb = NaiveLogisticRegression._get_ranked_values(nb_predictions)
		ranked_lr = NaiveLogisticRegression._get_ranked_values(lr_predictions)
		nb_above = NaiveLogisticRegression._get_num_above_half(nb_predictions)
		lr_above = NaiveLogisticRegression._get_num_above_half(lr_predictions)
		total_above_half = nb_above + lr_above
		predictions = list(initial_predictions)
		n_adjustments = 0
		for u in uncertain_idx:
			if nb_predictions[u] != lr_predictions[u]:
				nb_rank = ranked_nb[u]
				lr_rank = ranked_lr[u]
				predictions[u] = nb_rank + lr_rank >= total_above_half
				n_adjustments += 1
		return tuple(round(p) for p in predictions), n_adjustments

	@staticmethod
	def _predict_by_naive_bayes(
			uncertain_idx: Iterable[int],
			initial_predictions: Sequence[float],
			nb_predictions: Sequence[float]) -> Tuple[Tuple[int], int]:
		predictions = list(initial_predictions)
		n_adjustments = 0
		for u in uncertain_idx:
			predictions[u] = nb_predictions[u]
			n_adjustments += 1
		return tuple(round(p) for p in predictions), n_adjustments

	@staticmethod
	def _get_ranked_values(x: Iterable) -> Mapping[int, int]:
		ranked = enumerate(sorted(enumerate(x), key=lambda y: y[1]))
		# {index: rank}
		return {idx_and_val[0]: rank for rank, idx_and_val in ranked}

	@staticmethod
	def _get_num_above_half(x: Iterable) -> int:
		return sum(map(lambda v: v > 0.5, x))

	def save(self, file: str) -> NoReturn:
		pass

	@staticmethod
	def load(file: str):
		pass

	def get_name(self) -> str:
		return self.__class__.__name__
