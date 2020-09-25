import random
from collections import Counter
from typing import Collection, Union

from typing import Any, Iterable, SupportsFloat, SupportsInt, Union
import statistics
import numpy as np


def gain_ratio(feature, label) -> float:
	return information_gain(feature, label) / entropy(feature)


def information_gain(feature, label) -> float:
	return entropy(label) - conditional_entropy(label, feature)
from collections import deque
from P1 import algorithm
import copy
Number = Union[SupportsInt, SupportsFloat]


def conditional_entropy(event, given) -> float:
	return sum(
		expectation(g, entropy(conditional_probability(event, e, given, g)))
		for e in set(event) for g in set(given)
	)
#shuffle the training data into n blocks with a random number of additional units from other blocks
def shuffle_blocks(labels: Iterable, blocks: int):
    pos_lab = find_indices(labels, lambda y: y > 0)
    neg_lab = find_indices(labels, lambda y: y <= 0)
    pos_per_fold, pos_remainder = divmod(len(pos_lab), blocks)
    neg_per_fold, neg_remainder = divmod(len(neg_lab), blocks)

    random.shuffle(pos_lab)
    random.shuffle(neg_lab)
    data = []
    pos_lab_copy = copy.deepcopy(pos_lab)
    neg_lab_copy = copy.deepcopy(neg_lab)
    #how many additional examples we want not in the block per pos lab and neg lab
    additional_pos = round(pos_per_fold * blocks / 2)
    additional_neg =round(neg_per_fold * blocks / 2)


    for x in range(blocks):
        pos_count = pos_per_fold + min(1, pos_remainder)
        neg_count = neg_per_fold + min(1, neg_remainder)
        pos_remainder = max(0, pos_remainder - 1)
        neg_remainder = max(0, neg_remainder - 1)
        data[x] = [pos_lab[:pos_count] + neg_lab[:neg_count]]
        del pos_lab[:pos_count]
        del neg_lab[:neg_count]
        pos_pool = [x for x in pos_lab_copy if x not in data[x]] #get pool of positives and pool of negatives that aren't in data[x]
        neg_pool = [x for x in neg_lab_copy if x not in data[x]]
        pos_deck = list(range(len(pos_pool)))
        neg_deck = list(range(len(neg_pool)))
        random.shuffle(pos_deck)
        random.shuffle(neg_deck)
        data[x] = [data[x] + pos_deck[:additional_pos] + neg_deck[:additional_neg]]
    return data


def stochastic_information_gain(x: Iterable[Number], labels: Iterable[Any], partitions, use_ig):
    listed_indices = shuffle_blocks(labels, partitions)
    for i in range(listed_indices):
        subset_feats = x[listed_indices[i]]
        subset_labels = labels[listed_indices[i]]
        ig = []
        if use_ig:
            ig[i] = information_gain(subset_feats, subset_labels)
        else:
            ig[i] = gain_ratio(subset_feats, subset_labels)
    return statistics.mean(ig)

def find_indices(list, condition):
    return [i for i, elem in enumerate(list) if condition(elem)]


def gain_ratio(feature: Iterable[Any], label: Iterable[Any]) -> float:
    h_x = entropy(feature)
    return information_gain(feature, label) / h_x


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
