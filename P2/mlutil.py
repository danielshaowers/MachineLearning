import collections
import functools
import operator
from numbers import Number
from typing import Any, Callable, Collection, Dict, Generator, Iterable, \
	Sequence, Tuple, Union

import numpy as np

import mldata


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


def get_example_features(
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
		feature_types: Collection[mldata.Feature.Type] = None,
		as_tuple: bool = True,
		as_dict: bool = False,
		index_as_key: bool = False) -> Union[Tuple, Dict, Generator]:
	"""Retrieves all values on a per-feature basis.

	The examples of a feature can be considered a column of the ExampleSet.

	Args:
		data: ExampleSet to retrieve feature examples.
		start_index: Index of the starting feature. The default value of 1
			assumes the first feature of the ExampleSet is the identifier
			feature, which is not meaningful when performing classification.
		feature_types: Types of features to include. If None, both continuous
			and discrete types will be included.
		as_tuple: True will return the examples as a tuple. If as_dict is
			False, then the returned result will be a tuple of tuples,
			where each tuple is feature. If both as_tuple and as_dict are
			False, then what is returned is a generator of generators.
		as_dict: True will return a dictionary with the feature (object or
			index in schema, see index_as_key) as the key and the examples as
			the value. If as_tuple is True, the values will be a tuple,
			instead of a generator.
		index_as_key: If True, use the feature index in the ExampleSet schema,
			instead of the feature object.
	Returns:
		A tuple of tuples, generator of generators, dictionary of tuples,
		or dictionary of generators of examples, grouped by feature.
	"""
	# get_features_info() already accounts for removing ID feature
	f_info = get_features_info(data)[start_index - 1:]
	if feature_types is None:
		f_idx = {i + 1 for i, f in enumerate(f_info)}
	else:
		f_idx = {
			i + 1 for i, f in enumerate(f_info) if f.type in feature_types
		}
	examples = (
		(data[e][f] for e in range(len(data)) if f in f_idx)
		for f in range(start_index, len(data.schema) - 1)
	)
	if as_tuple:
		if as_dict:
			if index_as_key:
				examples = {
					i + 1: tuple(ex) for i, ex in enumerate(examples)
					if i + 1 in f_idx
				}
			else:
				examples = {f: tuple(ex) for f, ex in zip(f_info, examples)}
		else:
			examples = tuple(tuple(ex) for ex in examples)
	else:
		if as_dict:
			if index_as_key:
				examples = {
					i + 1: ex for i, ex in enumerate(examples)
					if i + 1 in f_idx
				}
			else:
				examples = {f: ex for f, ex in zip(f_info, examples)}
	return examples


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
	return collections.Counter(get_labels(data)).most_common(1)[0][0]


def print_label_ratio(data: mldata.ExampleSet):
	print(collections.Counter(get_labels(data)).most_common())


def get_feature_index(data: mldata.ExampleSet, feature: mldata.Feature) -> int:
	return data.schema.index(feature)


def get_label_info(data: mldata.ExampleSet) -> mldata.Feature:
	return data.schema[-1]


def quantify_nominals(data: np.array, types):
	indices = np.where(types == 'NOMINAL')[0]
	categories = [np.unique(data[i]) for i in indices]
	quantified = data
	for m, z in enumerate(indices):
		# find indices of each category
		val_idxs = [np.argwhere(cat == data[z]) for cat in categories[m]]
		for i, idxs in enumerate(val_idxs):
			for id in idxs:
				# avoids using 0, which is uninformative
				quantified[z][id[0]] = (i + 1) / len(val_idxs)
	return quantified


def convert_to_numpy(data: mldata.ExampleSet):
	info = get_features_info(data)
	types = np.array([n.type for n in info])
	np_data = np.array(data).transpose()
	np_data = np_data[1:len(np_data) - 1]
	return np_data, types


def compute_roc(scores, truths):
	predictions = np.array(scores)
	labels = np.array(truths)
	sorted_ind = np.argsort(scores)
	sorted_labels = labels[sorted_ind]
	sorted_scores = predictions[sorted_ind]
	thresholds, thresh_idxs = np.unique(sorted_scores, return_index=True)
	coordinates = np.zeros([len(thresholds) + 1, 2])
	best_point = [-1, -1]
	tot_p = np.sum(labels)  # total positive
	tot_n = len(labels) - tot_p  # track total negative
	thresh_idxs = np.append(thresh_idxs, len(labels), len(labels))

	tp = sum(sorted_labels)
	fp = len(sorted_labels) - tp
	for i in range(len(thresh_idxs) - 1):
		# just track tp, fp, and total pos, total neg
		curr_thresh = range(thresh_idxs[i], thresh_idxs[i + 1])
		tp -= np.sum(sorted_labels[curr_thresh])
		# only considering below thresh, so we only consider curr thresh
		thresh_idx_diff = thresh_idxs[i] - thresh_idxs[i + 1]
		fp += thresh_idx_diff + np.sum(sorted_labels[curr_thresh])
		coordinates[len(coordinates) - i - 2][0] = tp / tot_p  # tpr
		coordinates[len(coordinates) - i - 2][1] = fp / tot_n  # fpr
		precision = tp / (tp + fp)
		recall = tp / tot_p
		best_point[0] = precision / recall
		best_point[1] = thresholds[i]
	# next compute area underneath by trapezoidal area approximation
	auc = 0
	end_x = 1
	end_y = 1
	coordinates[len(coordinates) - 1] = [0, 0]  # edge case
	for coord in coordinates:
		start_x = coord[0]
		start_y = coord[1]
		# get area of trapezoidal region
		auc += (end_x - start_x) * (start_y + end_y) / 2
		end_x = start_x
		end_y = start_y
	return auc, best_point[1]


def prediction_stats(scores, truths, threshold=0.5):
	predictions = np.array(scores)
	labels = np.array(truths)
	predicted_labels = predictions >= threshold
	tp, tn, fp, fn = compute_tf_fp(predicted_labels, labels)
	accuracy = sum(predicted_labels == labels) / len(labels)
	precision = tp / max((tp + fp), 1)
	recall = tp / max((tp + fn), 1)
	specificity = tn / max(sum(labels == 0), 1)
	return accuracy, precision, recall, specificity, [tp, tn, fp, fn]


# compute true pos, false pos, true neg, false neg
def compute_tf_fp(predicted_labels: Sequence, truths: Sequence):
	labels = np.array(truths)
	pos_truths = np.where(labels > 0)[0]
	neg_truths = np.where(labels <= 0)[0]
	tp = sum(1 for e in pos_truths if predicted_labels[e] == 1)
	tn = sum(1 for i in neg_truths if predicted_labels[i] == 0)
	fp = sum(1 for e in neg_truths if predicted_labels[e] == 1)
	fn = sum(1 for e in pos_truths if predicted_labels[e] == 0)
	return tp, tn, fp, fn
