from abc import ABC, abstractmethod
from collections import deque, namedtuple

import numpy as np

import metrics
import mlddata_util as util
from Node import Node
from mldata import ExampleSet, Feature


# for continuous and nominal, we don't need to remove them after we come up
# with one test. we just remove if it's a pure node
# or track how many tests exist for nominal and continuous
# ask for clarification what these each are. what is features vs dataset? is
# dataset just a compilation of label and features?

class Model(ABC):

	def __init__(self):
		super(Model, self).__init__()

	@abstractmethod
	def train(self, data: ExampleSet):
		pass

	@abstractmethod
	def predict(self, data: ExampleSet):
		pass


class ID3(Model):
	def __init__(self, max_depth: int = None):
		self.max_depth = max_depth
		if max_depth < 1:
			self.grow_full_tree = True
		else:
			self.grow_full_tree = False
		self.model_metrics = dict()
		super(ID3, self).__init__()

	def train(self, data: ExampleSet) -> Node:
		return self.id3(data, Node())

	def predict(self, data: ExampleSet):
		pass

	def id3(self, dataset: ExampleSet, parent: Node, depth: int = 0):
		"""Generates a decision tree using the ID3 algorithm.

		Pseudo-code (from Machine Learning, Mitchell):
			1. Create a root node for the tree.
			2. If all examples in the dataset are of the same class, return
				the single-node tree with that class label.
			3. If features is empty, return the single-node tree with the
				target label that is most common among the observations.
			4. Otherwise:
				4.1 Let F be the feature in features that "best" classifies
					observations.
				4.2 Let root be F.
				4.3 For each possible value f_i of F:
					4.3.1 Add a new tree branch below root, corresponding to
						the test F = f_i.
					4.3.2 Let features_fi be the subset of features that have
						value f_i for F.
					4.3.3 If features_fi is empty, then below this new branch,
						add a leaf node with the most common value of target
						among the observations as the label.
					4.3.4 Otherwise, below this new branch, add the subtree
						id3(features_fi, target, features - {F}).
			Return root.

		Args:
			dataset: Collection of training examples.
			parent: Parent node in the decision tree.
			depth: Current depth of the decision tree.

		Returns:
			Decision tree that classifies the given observations.
		"""

		# TODO Other stopping criteria: at max_depth, no more features/tests
		if util.is_homogeneous(dataset):
			return Node(data=util.get_labels(dataset)[0], parent=parent)

	# 4.1 Get best feature and make it the root of the subtree
	@staticmethod
	def _get_best_feature(data: ExampleSet, use_info_gain: bool = True):
		labels = util.get_labels(data)
		features_examples = util.get_feature_examples(data)
		types = [feature.type for feature in util.get_features_info(data)]
		for f_examples, f_type in zip(features_examples, types):
			if f_type == Feature.Type.NOMINAL:
				if use_info_gain:
					split_criteria_val = metrics.info_gain(labels, f_examples)
				else:
					split_criteria_val = metrics.gain_ratio(labels, f_examples)
			else:
				partition = \
					ID3._get_best_partition(f_examples, labels, use_info_gain)

			feat_array = np.array(best_feat)
			lChild_idx = np.argwhere(feat_array <= partition)
			rChild_idx = numpy.argwhere(feat_array > partition)
			leftChild = parent_node.left(Node(parent_node, partition))
			rightChild = parent_node.right(Node(parent_node, partition))

	# 4.2 Make the best feature the root of the subtree

	# Alternative steps to the pseudo code above
	# 4.3 Find the best feature-value test for the given feature (
	# 		according to the split criteria)

	# 4.4 Partition the dataset based on this test

	# 4.5 If one partition is empty, assign it a leaf node that is the
	# 		label based on the current dataset (majority class)

	# 4.5 For the non empty partition, recurse

	# root = Node(bestF)a

	# find every time the class label changes
	@staticmethod
	def _get_best_partition(feature_examples, labels, use_info_gain: bool):
		ValueLabel = namedtuple('ValueLabel', ['value', 'label'])
		pairs = [ValueLabel(v, l) for v, l in zip(feature_examples, labels)]
		pairs = tuple(sorted(pairs, key=lambda p: p.value))
		splits = []
		prev = pairs[0]
		for i_pair in range(1, len(pairs)):
			pair = pairs[i_pair]
			if pair.label != prev.label:
				splits.append(pair.value)
			prev = pair

		info_gain = metrics.info_gain(labels, feature_examples)

		splits = deque()
		labelsN = np.array(labels)
		featN = np.array(feat)
		sorted_indices = np.argsort(featN)
		featN = np.sort(featN)
		labelsN = labelsN[sorted_indices]  # sort according to feature values
		prev_type = labelsN[0]
		splits.append(0)
		for x in range(1, len(labelsN)):
			if labelsN[x] != prev_type:
				# splits.append(featN(x)) #add to split if the type changes so
				# we don't check every single label
				splits.append(x)
			prev_type = labelsN[x]
		bestSplit = 0
		scoring = np.array(splits)
		cnt = 0;
		while splits:
			finish = splits.pop()
			split_feats = np.zeros(finish) + np.zeros(len(labelsN) - finish)
			if use_ig:
				# feed in the labels and
				scoring[cnt] = info_gain(split_feats, labelsN)  #
			else:
				scoring[cnt] = gain_ratio(split_feats, labelsN)
			cnt = cnt + 1
		return featN[np.argmax(scoring)]  # this returns the best split to use.

	# however...it doesn't show directionality

	def find_indices(list, condition):
		return [i for i, elem in enumerate(list) if condition(elem)]
