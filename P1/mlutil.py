import functools
import operator
from collections import Counter
from numbers import Number
from typing import Any, Callable, Iterable, Sequence, Set, Tuple

import mldata


def binarize_feature(values: Iterable, test: Callable[[Any], bool]) -> Tuple:
	return tuple(test(v) for v in values)


def create_all_split_tests(
		data: mldata.ExampleSet,
		drop_single_tests: bool = True) -> Tuple:
	types = (feature.type for feature in get_features_info(data))
	exs = get_feature_examples(data)
	labels = get_labels(data)
	tests = (create_split_tests(e, t, labels) for e, t in zip(exs, types))
	if drop_single_tests:
		tests = (t for t in tests if len(t) > 1)
	return tuple(tests)


def create_split_tests(
		values: Sequence,
		feature_type: mldata.Feature.Type,
		labels: Sequence = None) -> Tuple[Callable]:
	tests = tuple()
	categorical = {mldata.Feature.Type.NOMINAL, mldata.Feature.Type.BINARY}
	if feature_type in categorical:
		tests = create_discrete_split_tests(values)
	elif feature_type == mldata.Feature.Type.CONTINUOUS and labels is not None:
		split_values = find_split_values(values, labels)
		tests = create_continuous_split_tests(split_values)
	return tests


def create_discrete_split_tests(values: Iterable) -> Tuple:
	return tuple(functools.partial(operator.eq, v) for v in set(values))


def find_split_values(values: Sequence[Number], labels: Sequence) -> Set:
	ex = sorted(zip(values, labels), key=lambda x: x[0])
	return {ex[i][0] for i in range(1, len(ex)) if ex[i - 1][1] != ex[i][1]}


def create_continuous_split_tests(split_values: Iterable[Number]) -> Tuple:
	return tuple(functools.partial(operator.le, v) for v in set(split_values))


def is_homogeneous(data: mldata.ExampleSet) -> bool:
	return len(set(get_labels(data))) == 1


def get_majority_label(data: mldata.ExampleSet):
	return Counter(get_labels(data)).most_common()


def get_features(data: mldata.ExampleSet, index: int = None) -> Tuple:
	if index is None:
		features = tuple(tuple(example[1:-1] for example in data))
	else:
		features = tuple(data[index][1:-1])
	return features

# generate feature examples in a nested tuple. each index corresponds to one feature.
# exclude the truth label and the two id's
def get_feature_examples(data: mldata.ExampleSet, index: int = None) -> Tuple:
	if index is None:
		examples = tuple(
			tuple(data[e][f] for e in range(len(data)))
			for f in range(2, len(data.schema) - 1))
	else:
		examples = tuple(example[index] for example in data)
	return examples

#get data values only, excluding the first two indices in data
def exclude_schema(data: mldata.ExampleSet):
	new_data = []
	for i,d in enumerate(data):
		new_data.append(d[2:])
	return new_data


def get_labels(data: mldata.ExampleSet, index: int = None):
	if index is None:
		labels = tuple(example[-1] for example in data)
	else:
		labels = data[index][-1]
	return labels


def get_features_info(data: mldata.ExampleSet, index: int = None):
	if index is None:
		info = tuple(data.schema[1:-1])
	else:
		info = data.schema[index]
	return info


def get_feature_index(data: mldata.ExampleSet, feature: mldata.Feature) -> int:
	return data.schema.index(feature)


def get_label_info(data: mldata.ExampleSet) -> mldata.Feature:
	return data.schema[-1]
