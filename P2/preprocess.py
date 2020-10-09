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
		as_tuple=False,
		as_dict=True,
		index_as_key=True
	)
	if len(continuous_exs) == 0:
		return data
	standardized = {i: stats.zscore(exs) for i, exs in continuous_exs.items()}
	examples = [
		[
			standardized[f][e] if f in standardized else ex_val
			for f in range(len(data.schema))
		] for e, ex_val in enumerate(data)
	]
	example_set = mldata.ExampleSet(data.schema)
	example_set.append(examples)
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
		as_tuple=False,
		as_dict=True,
		index_as_key=True
	)
	if len(discrete_exs) == 0:
		return data
	near_zeros = {
		i for i, exs in discrete_exs.items() if discrete_var(exs) <= cut_off
	}
	examples = [
		[ex[f] for f in range(len(data.schema)) if f not in near_zeros]
		for ex in data
	]
	schema = [f for i, f in enumerate(data.schema) if i not in near_zeros]
	subset = mldata.ExampleSet(schema)
	subset.append(examples)
	return subset
