import functools
import operator
from collections import Counter
from numbers import Number
from typing import Any, Callable, Generator, Iterable, Tuple, Union

import mldata


def binarize_feature(
		values: Iterable,
		test: Callable[[Any], bool],
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	binarized = (test(v) for v in values)
	return tuple(binarized) if as_tuple else binarized


def create_all_split_tests(
		data: mldata.ExampleSet,
		drop_single_tests: bool = True,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	types = (feature.type for feature in get_features_info(data))
	exs = get_feature_examples(data, as_tuple=False)
	labels = get_labels(data, as_tuple=False)
	# TODO as_tuple = False and then tuple(tuple(t) for t in tests)
	tests = (create_split_tests(e, t, labels) for e, t in zip(exs, types))
	if drop_single_tests:
		tests = (t for t in tests if len(t) > 1)
	return tuple(tests) if as_tuple else tests


def create_split_tests(
		values: Iterable,
		feature_type: mldata.Feature.Type,
		labels: Iterable = None,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	tests = ()
	categorical = {mldata.Feature.Type.NOMINAL, mldata.Feature.Type.BINARY}
	if feature_type in categorical:
		tests = create_discrete_split_tests(values, as_tuple=False)
	elif feature_type == mldata.Feature.Type.CONTINUOUS and labels is not None:
		split_values = find_split_values(values, labels, as_tuple=False)
		tests = create_continuous_split_tests(split_values, as_tuple=False)
	return tuple(tests) if as_tuple else tests


def create_discrete_split_tests(
		values: Iterable,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	tests = (functools.partial(operator.eq, v) for v in set(values))
	return tuple(tests) if as_tuple else tests


def find_split_values(
		values: Iterable[Number],
		labels: Iterable,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	ex = sorted(zip(values, labels), key=lambda x: x[0])
	splits = (ex[i][0] for i in range(1, len(ex)) if ex[i - 1][1] != ex[i][1])
	return tuple(splits) if as_tuple else splits


def create_continuous_split_tests(
		split_values: Iterable,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	tests = (functools.partial(operator.le, v) for v in set(split_values))
	return tuple(tests) if as_tuple else tests


def is_homogeneous(data: mldata.ExampleSet) -> bool:
	return len(set(get_labels(data))) == 1


def get_majority_label(data: mldata.ExampleSet):
	return Counter(get_labels(data)).most_common()


def get_features(
		data: mldata.ExampleSet,
		example_index: int = None,
		feature_index: int = None,
		as_tuple: bool = True) -> Union[Tuple, Generator, Any]:
	if example_index is None:
		if feature_index is None:
			features = (example[1:-1] for example in data)
		else:
			features = (example[feature_index] for example in data)
		features = tuple(features) if as_tuple else features
	else:
		if feature_index is None:
			features = data[example_index][1:-1]
			features = tuple(features) if as_tuple else features
		else:
			features = data[example_index][feature_index]
	return features


# generate feature examples in a nested tuple. each index corresponds to one
# feature.
# exclude the truth label and the two id's
def get_feature_examples(
		data: mldata.ExampleSet,
		start_index: int = None,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	start_index = 1 if start_index is None else start_index
	examples = (
		(data[e][f] for e in range(len(data)))
		for f in range(start_index, len(data.schema) - 1))
	return tuple(tuple(ex) for ex in examples) if as_tuple else examples


def get_labels(
		data: mldata.ExampleSet,
		example_index: int = None,
		as_tuple: bool = True) -> Union[Tuple, Generator, Any]:
	if example_index is None:
		labels = (example[-1] for example in data)
		labels = tuple(labels) if as_tuple else labels
	else:
		labels = data[example_index][-1]
	return labels


def get_features_info(
		data: mldata.ExampleSet,
		feature_index: int = None,
		as_tuple: bool = True) -> Union[Tuple, Generator, mldata.Feature]:
	if feature_index is None:
		info = (data.schema[1:-1])
		info = tuple(info) if as_tuple else info
	else:
		info = data.schema[feature_index]
	return info


def get_feature_index(data: mldata.ExampleSet, feature: mldata.Feature) -> int:
	return data.schema.index(feature)


def get_label_info(data: mldata.ExampleSet) -> mldata.Feature:
	return data.schema[-1]
