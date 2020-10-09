import abc
from typing import Tuple

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
	def predict(self, data: mldata.ExampleSet) -> Tuple:
		pass
