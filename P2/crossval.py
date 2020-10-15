import itertools
import random
from collections import defaultdict
from typing import Sequence, Tuple

import mldata
import mlutil
import model


def cross_validate(
		learner: model.Model,
		data: mldata.ExampleSet,
		n_folds: int,
		save_as: str = None) -> Tuple[Tuple, Tuple]:
	"""Performs stratified cross validation on a general learner and data set.

	Args:
		learner: Model instance whose task is train on the data.
		data: Full set of training examples.
		n_folds: Number of folds to perform cross validation.
		save_as: Base name of the file. Will be appended with the fold
			iteration (if there are multiple folds) and .txt.

	Returns:
		Two tuples, the first being the predictions of from each fold,
		and then second being the corresponding labels.
	"""
	fold_predictions = []
	fold_test_labels = []
	if n_folds < 1:
		raise ValueError('Minimum number of folds is 1 (full dataset)')
	folds = get_folds(data, n_folds)
	for i in range(n_folds):
		if n_folds == 1:
			train = data
			test = data
		else:
			train, test = get_train_test_split(folds, i)
		learner.fold = i + min(n_folds, 2) - 1
		learner.train(train)
		if save_as:
			if n_folds == 1:
				learner.save(f'{save_as}.txt')
			else:
				learner.save(f'{save_as}_{i + 1}.txt')
		fold_predictions.append(learner.predict(test))
		fold_test_labels.append(mlutil.get_labels(test))
	return tuple(fold_predictions), tuple(fold_test_labels)


def get_train_test_split(folds: Sequence, test_fold_ind: int) -> Tuple:
	"""Creates the training and test sets from dataset folds.
	Args:
		folds: A sequence of sequences, each being a fold of the data.
		test_fold_ind: Index of the training set fold.

	Returns:
		A training set and test set.
	"""
	train_folds = [folds[i] for i in range(len(folds)) if i != test_fold_ind]
	training_examples = list(itertools.chain.from_iterable(train_folds))
	train_set = mldata.ExampleSet(training_examples)
	test_set = mldata.ExampleSet(folds[test_fold_ind])
	return train_set, test_set


def get_folds(dataset: mldata.ExampleSet, n_folds: int) -> Tuple:
	"""Creates n stratified folds from the dataset.

	For each unique value of the label, examples from the dataset are
	assigned to a fold round robin style. It may be the case that some folds
	have more training examples than others, depending  on the number of
	folds and number of examples in the dataset.

	Args:
		dataset: Collection of training examples on which to generate folds.
		n_folds: Number of folds to create.

	Returns:
		Tuple of lists, where each list is a fold of examples.
	"""
	folds = defaultdict(list)
	labels = mlutil.get_labels(dataset)
	for label in set(labels):
		examples = [e for e, lab in zip(dataset, labels) if lab == label]
		random.shuffle(examples)
		num_folds = max(1, n_folds)
		for i, example in enumerate(examples):
			folds[i % num_folds].append(example)
	return tuple(folds.values())
