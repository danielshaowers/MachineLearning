from collections import Counter
from typing import Any, Iterable, SupportsFloat, SupportsInt, Union

import numpy as np

Number = Union[SupportsInt, SupportsFloat]


def gain_ratio(feature: Iterable[Any], label: Iterable[Any]) -> float:
	pass


def information_gain(x: Iterable[Number], label: Iterable[Any]) -> float:
	pass


def conditional_entropy(x: Iterable[Number], cond: Iterable[Number]) -> float:
	pass


def entropy(x: Iterable[Number], base=2) -> float:
	frequencies = np.array(Counter(x).values())
	probabilities = frequencies / len(frequencies)
	return - expectation(numpy_log(probabilities, base), probabilities)


def expectation(x: Iterable[Number], probabilities: Iterable[Number]) -> float:
	return sum(np.array(x) * probabilities)


def numpy_log(x: Iterable[Number], base: Number) -> Union[float, np.ndarray]:
	return np.log(x) / np.log(base)
