import functools
import operator
import random
from collections import defaultdict
from typing import List, Sequence, Tuple

import mldata
import mlutil


def get_train_test_split(folds: Sequence[Sequence], test_fold_ind: int):
	train_folds = [folds[i] for i in range(len(folds)) if i != test_fold_ind]
	train_set = mldata.ExampleSet(functools.reduce(operator.add, train_folds))
	test_set = mldata.ExampleSet(folds[test_fold_ind])
	return train_set, test_set


def get_folds(dataset: mldata.ExampleSet, n_folds: int) -> Tuple[List]:
	folds = defaultdict(list)
	labels = mlutil.get_labels(dataset)
	for label in set(labels):
		examples = [e for e, lab in zip(dataset, labels) if lab == label]
		random.shuffle(examples)
		num_folds = max(1, n_folds)
		for i, example in enumerate(examples):
			folds[i % num_folds].append(example)
	return tuple(folds.values())
