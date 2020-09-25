from abc import ABC, abstractmethod
from collections import Counter, deque, namedtuple

import numpy as np

from .Node import Node
# TODO Fix this -- not good coding practice
from .metrics import gain_ratio, information_gain
from .mldata import Feature
from .observation import Observation, ObservationSet


# for continuous and nominal, we don't need to remove them after we come up
# with one test. we just remove if it's a pure node
# or track how many tests exist for nominal and continuous
# ask for clarification what these each are. what is features vs dataset? is
# dataset just a compilation of label and features?

def get_labels(dataset: ObservationSet):
	return [o.label for o in dataset]


def is_pure(dataset: ObservationSet) -> bool:
	return len(set(get_labels(dataset))) == 1


def get_majority_label(dataset: ObservationSet):
	label_counter = Counter([o.label for o in dataset])
	return label_counter.most_common()


class Model(ABC):

	@abstractmethod
	def train(self, data: ObservationSet):
		pass

	@abstractmethod
	def predict(self, observation: Observation):
		pass


class ID3(Model):
	def __init__(self, max_depth: int = None):
		self.max_depth = max_depth
		if max_depth < 1:
			self.grow_full_tree = True
		else:
			self.grow_full_tree = False
		self.model_metrics = dict()

	def train(self, dataset: ObservationSet):
		return self.id3(dataset, Node())

	def predict(self, observation: Observation):
		pass

	# TODO How do we keep track of the features we have tested? Per step 3 of
	#  the algorithm pseudo code, if we have tested all of our features,
	#  then we should just return the majority class. In our case,
	#  if we implicitly know what has been tested (vacuously true if the
	#  feature is not in the partitioned dataset), then we'll only get the
	#  case where we get a pure node?
	def id3(self, dataset: ObservationSet, parent: Node, depth: int = 0):
		"""
		Args:
			dataset: Collection of training examples.
			parent:

		Returns:
			A decision tree that classifies the given observations.

		ID3 pseudo-code (from Machine Learning textbook):
			1. Create a root node for the tree
			2. If all observations in the dataset are of the same class, return
				the single-node tree with that class label
			3. If features is empty, return the single-node tree with the
				target label that is most common among the observations
			4. Otherwise:
				4.1 Let F be the feature in features that "best" classifies
					observations
				4.2 Let root be F
				4.3 For each possible value f_i of F:
					4.3.1 Add a new tree branch below root, corresponding to
						the test F = f_i (this needs to be generalized to
						account for continuous features as well)
					4.3.2 Let features_fi be the subset of features that have
						value f_i for F
					4.3.3 If features_fi is empty, then below this new branch,
						add a leaf node with the most common value of target
						among the observations as the label
					4.3.4 Otherwise, below this new branch, add the subtree
						id3(features_fi, target, features - {F})
			Return root
		"""

		# TODO Other stopping criteria: at max_depth, no more features/tests
		if is_pure(dataset):
			return Node(data=get_labels(dataset)[0], parent=parent)

		# 4.1 Get best feature and make it the root of the subtree
		def get_best_feature(dataset: ObservationSet):
			features = dataset.to_features()
			for feat in features:

				if feat.Type == Feature.Type.NOMINAL:
					if igflag:
						ig = information_gain()
					else:
						ig = gain_ratio()
				else:
					partition = get_best_partition(feat, target, igflag)

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
	def get_best_partition(self, feature: Feature, label: Feature,
						   use_ig: bool):
		ValueLabelPair = namedtuple('ValueLabelPair', ['value', 'label'])
		feature_label_pairs = [
			ValueLabelPair(v, l) for v, l in zip(feature.values, label.values)
		]
		feature_label_pairs.sort(key=lambda p: p.value)
		splits = []
		prev = feature_label_pairs[0]
		for pair in feature_label_pairs:
			if pair.label != prev.label:
				splits.append(pair.value)

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
		while splits:  # could make this a dp problem. or greedy maybe? while
			# splits is not empty - it'd be faster to sort
			# feature first and get labels to correspond.
			finish = splits.pop()
			split_feats = np.zeros(finish) + np.zeros(len(labelsN) - finish)
			if use_ig:
				# feed in the labels and
				scoring[cnt] = information_gain(split_feats, labelsN)  #
			else:
				scoring[cnt] = gain_ratio(split_feats, labelsN)
			cnt = cnt + 1
		return featN[np.argmax(scoring)]  # this returns the best split to use.

	# however...it doesn't show directionality

	def find_indices(list, condition):
		return [i for i, elem in enumerate(list) if condition(elem)]
