import functools
import operator
import random
from collections import defaultdict
from typing import List

from .observation import Observation, ObservationSet

"""
1. get_folds
2. Create n train-test splits
3. Train and evaluate with each split, recording performance of each
"""


def cross_validate():
	pass


def get_train_test_split(folds: List[List[Observation]], test_fold_ind: int):
	train_folds = [folds[i] for i in range(len(folds)) if i != test_fold_ind]
	train_set = ObservationSet(functools.reduce(operator.add, train_folds))
	test_set = ObservationSet(folds[test_fold_ind])
	return train_set, test_set


def get_folds(dataset: ObservationSet, n_folds: int) -> List[List[Observation]]:
	# Handles improper n_folds value by defaulting to 1 fold
	num_folds = max(1, n_folds)
	# Key = fold index, value = observations in the fold
	folds = defaultdict(list)
	# Get all unique labels in the dataset
	labels = set(o.label for o in dataset)
	for label in labels:
		# Get only the observations that have a certain label
		observations = [o for o in dataset if o.label == label]
		random.shuffle(observations)
		# Add each observation a fold
		for i, observation in enumerate(observations):
			i_fold = i % num_folds
			folds[i_fold].append(observation)
	return list(folds.values())

# def cross_val(labels: Iterable, folds: int):
# 	pos_lab = find_indices(labels, lambda x: x > 0)
# 	neg_lab = find_indices(labels, lambda x: x <= 0)
# 	pos_per_fold, pos_remainder = divmod(len(pos_lab), folds)
# 	neg_per_fold, neg_remainder = divmod(len(neg_lab), folds)
#
# 	pos_count = pos_per_fold + min(1, pos_remainder)
# 	neg_count = neg_per_fold + min(1, neg_remainder)
# 	pos_remainder = max(0, pos_remainder - 1)
# 	neg_remainder = max(0, neg_remainder - 1)
#
# 	random.shuffle(pos_lab)
# 	random.shuffle(neg_lab)
# 	data = []
#
# 	for x in range(folds):
# 		data[x] = [pos_lab[:pos_count] + neg_lab[:neg_count]]
# 		del pos_lab[:pos_count]
# 		del neg_lab[:neg_count]
# 	return data
