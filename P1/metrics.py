from collections import Counter
from typing import Collection, Union

import numpy as np


def gain_ratio(feature, label) -> float:
	return information_gain(feature, label) / entropy(feature)


def information_gain(feature, label) -> float:
	return entropy(label) - conditional_entropy(label, feature)


def conditional_entropy(event, given) -> float:
	return sum(
		expectation(g, entropy(conditional_probability(event, e, given, g)))
		for e in set(event) for g in set(given)
	)


def conditional_probability(event, event_val, given, given_val) -> float:
	unconditional = probability(given, given_val)
	joint_freq = sum(
		1 for e, g in zip(event, given) if e == event_val and g == given_val
	)
	joint = joint_freq / len(min(event, given))
	return joint / unconditional


def entropy(probability, base=2) -> float:
	return - expectation(numpy_log(probability, base), probability)


def probability(x: Collection, val=None) -> Union[float, Collection[float]]:
	frequency = Counter(x)
	if val is not None:
		frequency = frequency[val]
	else:
		frequency = np.array(frequency.values())
	return frequency / len(x)


def expectation(x, probability) -> float:
	return sum(np.array(x) * np.array(probability))


def numpy_log(x, base) -> Union[float, Collection[float]]:
	return np.log(x) / np.log(base)
