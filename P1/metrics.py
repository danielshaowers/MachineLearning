from typing import Any, Iterable

from collections import Counter
import numpy as np
import math


def gain_ratio(feature: Iterable[Any], label: Iterable[Any]) -> float:
	pass


def information_gain(feature: Iterable[Any], label: Iterable[Any]) -> float:
	pass


def entropy(x: Iterable[Any], base=2) -> float:
	freqs = np.array(Counter(x).values())
	probs = freqs / freqs.size()
	log_probs = np.log(probs) / np.log(base)
	return -sum(probs * log_probs)


def prob_by_freq(feature: Iterable[Any], feature_val: Any,
				 label: Iterable[Any], label_val: Any):
	pass
