from typing import Tuple

from mldata import ExampleSet


def get_features(example_set: ExampleSet, example_index: int = None) -> Tuple:
	if example_index is None:
		features = tuple(tuple(example[1:-1] for example in example_set))
	else:
		features = tuple(example_set[example_index][1:-1])
	return features


def get_feature_examples(example_set: ExampleSet, index: int) -> Tuple:
	return tuple(example[index] for example in example_set)


def get_labels(example_set: ExampleSet, index: int = None):
	if index is None:
		labels = tuple(example[-1] for example in example_set)
	else:
		labels = example_set[index][-1]
	return labels


def get_features_info(example_set: ExampleSet, index: int = None):
	if index is None:
		info = tuple(example_set.schema[1:])
	else:
		info = example_set.schema[index]
	return info
