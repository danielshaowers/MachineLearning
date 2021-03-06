import functools
import operator
from collections import Counter
from numbers import Number
from typing import Any, Callable, Generator, Iterable, Tuple, Union

import mldata

"""
DESIGN PHILOSOPHY: Each utility function that can return an iterable should 
provide the functionality to either return a tuple or a generator. The former 
is preferred over a list for its immutability. The latter is useful in cases 
where the returned iterable is only needed for a single operation because it 
use much more memory efficient than a standard tuple or list.

Use type hints for parameters and return types, except when it is a nested 
sequence, collection, iterable, etc. These help with static type checking 
when developing and improve readability.
"""


def binarize_feature(
		values: Iterable,
		test: Callable[[Any], bool],
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	"""Given an iterable, return a binarized version.

	Trues in the returned iterable indicate those values satisfy the provided
	test. Otherwise, Falses indicate values that do not satisfy the test.

	Args:
		values: Values to binarize.
		test: Test on which to evaluate the values.
		as_tuple: True will return the iterable as a tuple, and as a
			generator otherwise.

	Returns:
		A tuple or generator of binarized values.
	"""
	binarized = (test(v) for v in values)
	return tuple(binarized) if as_tuple else binarized


def create_all_split_tests(
		data: mldata.ExampleSet,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	types = (feature.type for feature in get_features_info(data))
	exs = get_feature_examples(data, as_tuple=False)
	labels = get_labels(data)
	tests = (create_split_tests(e, t, labels) for e, t in zip(exs, types))
	return tuple(tuple(t) for t in tests) if as_tuple else tests


def create_split_tests(
		values: Iterable,
		feature_type: mldata.Feature.Type,
		labels: Iterable = None,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	tests = ()
	categorical = {
		mldata.Feature.Type.NOMINAL,
		mldata.Feature.Type.BINARY,
		mldata.Feature.Type.CLASS}
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


def get_features(
		data: mldata.ExampleSet,
		example_index: int = None,
		feature_index: int = None,
		as_tuple: bool = True) -> Union[Tuple, Generator, Any]:
	"""Retrieves the values of one or more examples and or more features.

	If neither index is specified, then the values of all of the features,
	with the exception of the first and  last features which are assumed to
	be the identifier and class label, for all of the examples are returned.

	If only the example_index is specified, then the values of all of the
	feature corresponding example are returned, with the exception of the
	first and  last features which are assumed to be the identifier and class
	label.

	If only the feature index is specified, then the value of corresponding
	feature is provided for all examples.

	Args:
		data: ExampleSet in which to retrieve feature values.
		example_index: Index of the example of interest.
		feature_index: Index of the feature of interest.
		as_tuple: True will return the iterable as a tuple, and as a
			generator otherwise.

	Returns:
		A tuple or generator if neither or only one of the indices are
		provided; otherwise, a single feature value.
	"""
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


def get_feature_examples(
		data: mldata.ExampleSet,
		start_index: int = 1,
		as_tuple: bool = True) -> Union[Tuple, Generator]:
	"""Retrieves all values on a per-feature basis.

	The examples of a feature can be considered a column of the ExampleSet.

	Args:
		data: ExampleSet to retrieve feature examples.
		start_index: Index of the starting feature. The default value of 1
			assumes the first feature of the ExampleSet is the identifier
			feature, which is not meaningful when performing classification.
		as_tuple: True will return the iterable as a tuple, and as a
			generator otherwise.

	Returns:
		A tuple of tuples or generator of generators in which each represents
		all the values of a given feature.
	"""
	examples = (
		(data[e][f] for e in range(len(data)))
		for f in range(start_index, len(data.schema) - 1))
	return tuple(tuple(ex) for ex in examples) if as_tuple else examples


def get_labels(
		data: mldata.ExampleSet,
		example_index: int = None,
		as_tuple: bool = True) -> Union[Tuple, Generator, Any]:
	"""Retrieves the value(s) of the label for one or more examples.

	Args:
		data: ExampleSet containing the label values.
		example_index: Index of the example for which the label value should
			be retrieved.
		as_tuple: True will return the iterable as a tuple, and as a
			generator otherwise.

	Returns:
		Either a tuple or generator of no example_index is supplied of the
		label value per example; or the single value of the label for the
		index at the supplied index.
	"""
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
	"""Returns the name, type, and values information for all features that
	are not of type ID and CLASS. It assumed that the first column of the
	data is the ID feature and the last column is the CLASS feature.

	Args:
		data: ExampleSet containing the feature information.
		feature_index: Index of the feature in the data.
		as_tuple: True will return the iterable as a tuple, and as a
			generator otherwise.

	Returns:
		A tuple or generator of the feature information.
	"""
	if feature_index is None:
		info = (data.schema[1:-1])
		info = tuple(info) if as_tuple else info
	else:
		info = data.schema[feature_index]
	return info


def is_homogeneous(data: mldata.ExampleSet) -> bool:
	return len(set(get_labels(data))) == 1


def get_majority_label(data: mldata.ExampleSet):
	return Counter(get_labels(data)).most_common(1)[0][0]

def print_label_ratio(data: mldata.ExampleSet):
	print(Counter(get_labels(data)).most_common())

def get_feature_index(data: mldata.ExampleSet, feature: mldata.Feature) -> int:
	return data.schema.index(feature)


def get_label_info(data: mldata.ExampleSet) -> mldata.Feature:
	return data.schema[-1]
