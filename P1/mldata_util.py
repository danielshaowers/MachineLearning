import functools
import operator
from collections import Counter
from typing import Iterable, Tuple

from mldata import ExampleSet, Feature


# TODO Account for continuous variables
def create_all_split_tests(data: ExampleSet, drop_single_tests=True) -> Tuple:
	types = [f.type for f in get_features_info(data)]
	examples = get_feature_examples(data)
	tests = tuple(create_split_tests(e, t) for e, t in zip(examples, types))
	if drop_single_tests:
		tests = tuple(t for t in tests if len(t) > 1)
	return tests


def create_split_tests(values: Iterable, feature_type: Feature.Type) -> Tuple:
	if feature_type in {Feature.Type.NOMINAL, Feature.Type.BINARY}:
		tests = tuple(functools.partial(operator.eq, v) for v in set(values))
	elif feature_type == Feature.Type.CONTINUOUS:
		tests = tuple(functools.partial(operator.le, v) for v in values)
	else:
		tests = tuple()
	return tests


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


def get_feature_examples(data: ExampleSet, index: int = None) -> Tuple:
	if index is None:
		examples = tuple(
			tuple(data[e][f] for e in range(len(data)))
			for f in range(1, len(data.schema) - 1)
		)
	else:
		examples = tuple(example[index] for example in data)
	return examples


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
