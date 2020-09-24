from collections import Counter
from typing import Any, Iterable, SupportsFloat, SupportsInt, Union

import numpy as np
from collections import deque
from P1 import algorithm

Number = Union[SupportsInt, SupportsFloat]


def gain_ratio(feature: Iterable[Any], ig: Number) -> float:
    h_x = entropy(feature)
    return ig / h_x


def information_gain(x: Iterable[Number], label: Iterable[Any]) -> float:
    h_y = entropy(label)
    uniqueVals = set(x)
    all_entropies = 0
    for val in uniqueVals:
        x_prob, h_y_x = conditional_entropy(x, val, label)
        all_entropies = all_entropies + x_prob * h_y_x
    return h_y - all_entropies

# h(y|X=condition)
# x is a feature vector, y is the labels, cond is the specific value of x
def conditional_entropy(x: Iterable[Number], cond: Iterable[Number], labels) -> float:
    indices = algorithm.find_indices(x, lambda val: val == cond)
    count = 0
    for idx in indices:
        if labels(idx) == 1:
            count = count + 1
    probability = count / len(indices)
    x_prob = len(indices) / len(labels)
    return x_prob, -probability * (numpy_log(probability, 2)) - (1 - probability) * numpy_log(1 - probability, 2)


def entropy(x: Iterable[Number], base=2) -> float:
    frequencies = np.array(Counter(x).values())
    probabilities = frequencies / len(frequencies)
    return - expectation(numpy_log(probabilities, base), probabilities)


def expectation(x: Iterable[Number], probabilities: Iterable[Number]) -> float:
    return sum(np.array(x) * probabilities)


def numpy_log(x: Iterable[Number], base: Number) -> Union[float, np.ndarray]:
    return np.log(x) / np.log(base)
