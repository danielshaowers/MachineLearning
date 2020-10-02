import enum
from abc import ABC, abstractmethod
from typing import Any, Callable, NoReturn, Tuple

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

	class Metrics(str, enum.Enum):
		FIRST_FEATURE = 'first_feature'
		MAX_DEPTH = 'max_depth'
		TREE_SIZE = 'tree_size'

	def __init__(self, max_depth: int = 1, split_function: Callable = None):
		self.max_depth = max_depth
		if split_function is None:
			self.split_function = metrics.info_gain  # we assign the function
		else:
			self.split_function = split_function
		self.model_metrics = dict()
		self.model = None
		super(ID3, self).__init__()

	def train(self, data: mldata.ExampleSet) -> NoReturn:
		self.model = self.id3(data)
		self._get_model_metrics()

	# return the relevant metrics to be reported
	def _get_model_metrics(self) -> NoReturn:
		if self.model is None:
			return
		self.model_metrics[ID3.Metrics.FIRST_FEATURE] = self.model.data.feature
		self.model_metrics[ID3.Metrics.MAX_DEPTH] = self.model.get_max_depth()
		self.model_metrics[ID3.Metrics.TREE_SIZE] = self.model.get_tree_size()

	# return predicted labels given an example set using this tree's class
	def predict(self, data: mldata.ExampleSet) -> Tuple:
		if self.model is None:
			predictions = tuple()
		else:
			predictions = tuple(
				self._predict_example(mldata.ExampleSet([example]), self.model)
				for example in data)
		return predictions

	# recursively predicts the label of a single example
	def _predict_example(self, example: mldata.ExampleSet, model: node.Node):
		if model.is_leaf():  # base condition. returns true or false
			return model.data
		idx = mlutil.get_feature_index(example, model.data.feature)
		# returns the specific feature value
		val = mlutil.get_features(example, example_index=0, feature_index=idx)
		# result of the test at this node
		result: bool = model.data.evaluate(val)
		if result:
			predicted = self._predict_example(example, model.left)
		else:
			predicted = self._predict_example(example, model.right)
		return predicted

	# main id3 method to build a decision tree recursively
	def id3(
			self,
			data: mldata.ExampleSet,
			parent: node.Node = None,
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
		current_node = node.Node(parent=parent)
		current_node.data = Test(feature=feature, test=test)
		left_data, right_data = self._partition_data(data, feature, test)
		if len(left_data) == 0:
			left_child = node.Node(data=majority_label, parent=current_node)
		else:
			left_child = self.id3(left_data, current_node, depth + 1)
		if len(right_data) == 0:
			right_child = node.Node(data=majority_label, parent=current_node)
		else:
			right_child = self.id3(right_data, current_node, depth + 1)
		current_node.left = left_child
		current_node.right = right_child
		return current_node

	def _get_best_feature_and_test(self, data: mldata.ExampleSet) -> Tuple:
		labels = mlutil.get_labels(data)
		feature_exs = mlutil.get_feature_examples(data)
		split_tests = mlutil.create_all_split_tests(data)
		l_type = mlutil.get_label_info(data).type
		label_tests = mlutil.create_split_tests(labels, l_type, as_tuple=False)
		# finds the information gain or gain ratio of each test
		split_values = [[
			self.split_function(labels, label_tests, f, [t])
			# get the tests for the ith feature
			for t in split_tests[i]] for i, f in enumerate(feature_exs)]
		i_max_feature = int(np.argmax([max(v) for v in split_values]))
		i_max_test = np.argmax(split_values[i_max_feature])
		best_test = split_tests[i_max_feature][i_max_test]
		# ID feature not considered when generating feature tests (add 1)
		best_feature = data.schema[i_max_feature + 1]
		return best_feature, best_test

	@staticmethod
	def _partition_data(
			data: mldata.ExampleSet,
			feature: mldata.Feature,
			test: Callable[[Any], bool]) -> Tuple:
		idx = mlutil.get_feature_index(data, feature)
		left_data = mldata.ExampleSet([e for e in data if test(e[idx])])
		right_data = mldata.ExampleSet([e for e in data if not test(e[idx])])
		return left_data, right_data

	def _at_max_depth(self, depth: int) -> bool:
		return False if self.max_depth < 1 else depth >= self.max_depth


class Test:
	def __init__(self, feature: mldata.Feature, test: Callable):
		self.feature = feature
		self._test = test

	def evaluate(self, value):
		return self._test(value)
