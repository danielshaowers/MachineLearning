import copy
import random
import statistics
from collections import Counter
from typing import Collection
from typing import Iterable, Union

import numpy as np


def gain_ratio(event, given) -> float:
	return info_gain(event, given) / entropy(given)


def info_gain(event, given) -> float:
	return entropy(event) - conditional_entropy(event, given)


# TODO Modify to pass in conditional tests of given values
def conditional_entropy(event, given, given_conds) -> float:
	return sum(
		expectation(g, entropy(conditional_probability(event, e, given, g)))
		for e in set(event) for g in set(given)
	)


# shuffle the training data into n blocks with a random number of additional
# units from other blocks
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
	# how many additional examples we want not in the block per pos lab and
	# neg lab
	additional_pos = round(pos_per_fold * blocks / 2)
	additional_neg = round(neg_per_fold * blocks / 2)

	for x in range(blocks):
		pos_count = pos_per_fold + min(1, pos_remainder)
		neg_count = neg_per_fold + min(1, neg_remainder)
		pos_remainder = max(0, pos_remainder - 1)
		neg_remainder = max(0, neg_remainder - 1)
		data[x] = [pos_lab[:pos_count] + neg_lab[:neg_count]]
		del pos_lab[:pos_count]
		del neg_lab[:neg_count]
		pos_pool = [x for x in pos_lab_copy if x not in data[x]]  # get pool
		# of positives and pool of negatives that aren't in data[x]
		neg_pool = [x for x in neg_lab_copy if x not in data[x]]
		pos_deck = list(range(len(pos_pool)))
		neg_deck = list(range(len(neg_pool)))
		random.shuffle(pos_deck)
		random.shuffle(neg_deck)
		data[x] = [
			data[x] + pos_deck[:additional_pos] + neg_deck[:additional_neg]]
	return data


def stochastic_information_gain(x, labels, partitions, use_ig):
	listed_indices = shuffle_blocks(labels, partitions)
	for i in range(listed_indices):
		subset_feats = x[listed_indices[i]]
		subset_labels = labels[listed_indices[i]]
		ig = []
		if use_ig:
			ig[i] = info_gain(subset_labels, subset_feats)
		else:
			ig[i] = gain_ratio(subset_labels, subset_feats)
	return statistics.mean(ig)


def find_indices(list, condition):
	return [i for i, elem in enumerate(list) if condition(elem)]


def conditional_probability(event, event_cond, given, given_cond) -> float:
	unconditional = probability(given, given_cond)
	joint_freq = sum(
		1 for e, g in zip(event, given) if event_cond(e) and given_cond(g)
	)
	joint = joint_freq / len(min(event, given))
	return joint / unconditional


def entropy(probability, base=2) -> float:
	return - expectation(numpy_log(probability, base), probability)


def probability(x: Collection, cond=None) -> Union[float, Collection[float]]:
	if cond is not None:
		frequency = sum(1 for y in x if cond(y))
	else:
		frequency = np.array(Counter(x).values())
	return frequency / len(x)


def expectation(x, probability) -> float:
	return sum(np.array(x) * np.array(probability))


def numpy_log(x, base) -> Union[float, Collection[float]]:
	return np.log(x) / np.log(base)
