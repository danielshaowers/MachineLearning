import statistics
from typing import Sequence

from scipy.stats import stats

import mldata
import mlutil

BINARY = mldata.Feature.Type.BINARY
CONTINUOUS = mldata.Feature.Type.CONTINUOUS
NOMINAL = mldata.Feature.Type.NOMINAL


def standardize(data: mldata.ExampleSet) -> mldata.ExampleSet:
	"""Standardizes (center and scales) all continuous features.
	Args:
		data: Collection of examples to pre-process.

	Returns:
		An ExampleSet with standardized continuous features.
	"""
	continuous_exs = mlutil.get_feature_examples(
		data=data,
		feature_types={CONTINUOUS},
		as_dict=True,
		index_as_key=True
	)
	if len(continuous_exs) == 0:
		return data
	standardized = {i: stats.zscore(exs) for i, exs in continuous_exs.items()}
	examples = []
	for e, ex_val in enumerate(data):
		example = mldata.Example(data.schema)
		# f is the feature number; e is the example index
		example.features = [
			standardized[f][e] if f in standardized else ex_val[f]
			for f in range(len(data.schema))
		]
		examples.append(example)
	example_set = mldata.ExampleSet(data.schema)
	example_set.extend(examples)
	return example_set


def remove_near_zero_variance(
		data: mldata.ExampleSet,
		cut_off: float = 0.1) -> mldata.ExampleSet:
	"""Removes all discrete features that have "low" variance.
	Args:
		data: Collection of examples to pre-process.
		cut_off: Features with a variance below this value will be removed
			from the data.

	Returns:
		Filtered data that does not include near-zero variance features.
	"""

	def discrete_var(values: Sequence) -> float:
		encoding = {k: v for v, k in enumerate(set(values))}
		return statistics.variance(encoding[v] for v in values)

	discrete_exs = mlutil.get_feature_examples(
		data=data,
		feature_types={BINARY, NOMINAL},
		as_dict=True,
		index_as_key=True
	)
	if len(discrete_exs) == 0:
		return data
	near_zeros = {
		i for i, exs in discrete_exs.items() if discrete_var(exs) <= cut_off
	}
	enumerated_subset_schema = [
		(i, f) for i, f in enumerate(data.schema) if i not in near_zeros
	]
	subset_schema = [feature for _, feature in enumerated_subset_schema]
	examples = []
	for ex in data:
		example = mldata.Example(subset_schema)
		example.features = [ex[i] for i, _ in enumerated_subset_schema]
		examples.append(example)
	subset = mldata.ExampleSet()
	subset.extend(examples)
	return subset
