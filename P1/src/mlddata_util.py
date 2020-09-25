from collections import Counter
from typing import Tuple

from mldata import ExampleSet


def is_homogeneous(data: ExampleSet) -> bool:
	return len(set(get_labels(data))) == 1


def get_majority_label(data: ExampleSet):
	return Counter(get_labels(data)).most_common()


def get_features(data: ExampleSet, example_index: int = None) -> Tuple:
	if example_index is None:
		features = tuple(tuple(example[1:-1] for example in data))
	else:
		features = tuple(data[example_index][1:-1])
	return features


def get_feature_examples(data: ExampleSet, index: int) -> Tuple:
	return tuple(example[index] for example in data)


def get_labels(data: ExampleSet, index: int = None):
	if index is None:
		labels = tuple(example[-1] for example in data)
	else:
		labels = data[index][-1]
	return labels


def get_features_info(data: ExampleSet, index: int = None):
	if index is None:
		info = tuple(data.schema[1:-1])
	else:
		info = data.schema[index]
	return info


def get_label_info(data: ExampleSet):
	return data.schema[-1]
