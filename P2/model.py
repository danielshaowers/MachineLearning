import abc
import dataclasses
from typing import Any

import mldata


class Model(abc.ABC):
	"""An abstract machine learning model.

	A model is trainable on a given ExampleSet and can subsequently predict
	the labels of new examples. Note that it is not possible to predict new
	examples prior to training the model.
	"""

	def __init__(self):
		super(Model, self).__init__()

	@abc.abstractmethod
	def train(self, data: mldata.ExampleSet):
		pass

	@abc.abstractmethod
	def predict(self, data: mldata.ExampleSet):
		pass


#@dataclasses.dataclass(frozen=True)
class Prediction:
	truth: Any
	confidence: float

	def __init__(self, value, confidence):
		self.truth = value
		self.confidence = confidence

	def __post_init__(self):
		if self.confidence < 0 or 1 < self.confidence:
			raise ValueError('Confidence must be between 0 and 1, inclusive')
