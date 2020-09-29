from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, NoReturn, Tuple

import numpy as np

import metrics
import mldata
import mlutil
import node


class Model(ABC):
	"""An abstract machine learning model.

	A model is trainable on a given ExampleSet and can subsequently predict
	the labels of new examples. Note that it is not possible to predict new
	examples prior to training the model.
	"""

	def __init__(self):
		super(Model, self).__init__()

	@abstractmethod
	def train(self, data: mldata.ExampleSet):
		pass

	@abstractmethod
	def predict(self, data: mldata.ExampleSet):
		pass


class ID3(Model):
	"""Decision tree classifier using the ID3 algorithm.

	"""

	class Metrics(Enum):
		FIRST_FEATURE = 'first_feature'
		MAX_DEPTH = 'max_depth'
		TREE_SIZE = 'tree_size'

	def __init__(self, max_depth: int = 1, split_function: Callable = None):
		self.max_depth = max_depth
		if split_function is None:
			self.split_function = metrics.info_gain
		else:
			self.split_function = split_function
		self.model_metrics = dict()
		self.model = None
		super(ID3, self).__init__()

	def train(self, data: mldata.ExampleSet) -> NoReturn:
		self.model = self.id3(data, node.Node())
		self._get_model_metrics()

	def _get_model_metrics(self) -> NoReturn:
		if self.model is None:
			return
		self.model_metrics[ID3.Metrics.FIRST_FEATURE] = self.model.data.feature
		self.model_metrics[ID3.Metrics.MAX_DEPTH] = self.model.get_max_depth()
		self.model_metrics[ID3.Metrics.TREE_SIZE] = self.model.get_tree_size()

	def predict(self, data: mldata.ExampleSet) -> Tuple:
		if self.model is None:
			predictions = tuple()
		else:
			predictions = tuple(
				self._predict_example(mldata.ExampleSet([example]), self.model)
				for example in data)
		return predictions

	def _predict_example(self, example: mldata.ExampleSet, root: node.Node):
		if root.is_leaf():
			return root.data
		index = mlutil.get_features_info(example).index(root.data.feature)
		feature_value = mlutil.get_features(example)[index]
		result: bool = root.data.evaluate(feature_value)
		if result:
			predicted = self._predict_example(example, root.left)
		else:
			predicted = self._predict_example(example, root.right)
		return predicted

	def id3(
			self,
			data: mldata.ExampleSet,
			parent: node.Node,
			depth: int = 0) -> node.Node:
		"""Generates a decision tree using the ID3 algorithm.

			Args:
				data: Collection of training examples.
				parent: Parent node in the decision tree.
				depth: Current depth of the decision tree.

			Returns:
				Decision tree that classifies the given observations.
			"""
		majority_label = mlutil.get_majority_label(data)
		if mlutil.is_homogeneous(data) or self._at_max_depth(depth):
			return node.Node(data=majority_label, parent=parent)
		feature, test = self._get_best_feature_and_test(data)
		parent.data = Test(feature=feature, test=test)
		left_data, right_data = self._partition_data(data, feature, test)
		if len(left_data) == 0:
			left_child = node.Node(data=majority_label, parent=parent)
		else:
			left_child = self.id3(left_data, parent, depth + 1)
		parent.left = left_child
		if len(right_data) == 0:
			right_child = node.Node(data=majority_label, parent=parent)
		else:
			right_child = self.id3(right_data, parent, depth + 1)
		parent.right = right_child
		return parent

	def _get_best_feature_and_test(self, data: mldata.ExampleSet) -> Tuple:
		labels = mlutil.get_labels(data)
		features = mlutil.get_feature_examples(data)
		# TODO Would be more efficient to calculate once and pass down
		split_tests = mlutil.create_all_split_tests(data)
		split_values = [
			# TODO Update parameters
			[self.split_function(labels, f, t) for t in split_tests[i]]
			for i, f in enumerate(features)]
		i_max_feature = int(np.argmax([max(v) for v in split_values]))
		i_max_test = np.argmax(split_values[i_max_feature])
		best_test = split_tests[i_max_feature][i_max_test]
		best_feature = data.schema[i_max_feature - 1]
		return best_feature, best_test

	@staticmethod
	def _partition_data(
			data: mldata.ExampleSet,
			feature: mldata.Feature,
			test: Callable) -> Tuple:
		index = mlutil.get_features_info(data).index(feature)
		left_data = mldata.ExampleSet([e for e in data if test(e[index])])
		right_data = mldata.ExampleSet([e for e in data if not test(e[index])])
		return left_data, right_data

	def _at_max_depth(self, depth: int) -> bool:
		if self.max_depth < 1:
			at_max = False
		else:
			at_max = depth == self.max_depth
		return at_max


class Test:
	def __init__(self, feature: mldata.Feature, test: Callable):
		self.feature = feature
		self._test = test

	def evaluate(self, value):
		return self._test(value)
