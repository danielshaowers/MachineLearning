import collections
import copy
import math
import random
import statistics
from typing import Any, Callable, Collection, Mapping
from typing import Iterable, Union


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


def gain_ratio(
		event: Collection,
		event_tests: Collection[Callable],
		given: Collection,
		given_tests: Collection[Callable]) -> float:
	return info_gain(event, event_tests, given, given_tests) / entropy(given)


def info_gain(
		event: Collection,
		event_tests: Collection[Callable],
		given: Collection,
		given_tests: Collection[Callable]) -> float:
	cond_entropy = conditional_entropy(event, event_tests, given, given_tests)
	# find the pure H(Y)
	h_y = sum([entropy(probability(event, e)) for e in event_tests])
	gain = h_y - cond_entropy
	return 0 if math.isnan(gain) else gain


# todo: modify so we cover both probabilities of x values.
def conditional_entropy(
		event: Collection,
		event_tests: Collection[Callable],  # y==1
		given: Collection,
		given_tests: Collection[Callable]) -> float:
	# todo: if it's not nominal, add this new function to a collection
	return sum(
		probability(given, g) * entropy(probability(event, e, given, g))
		for e in event_tests for g in given_tests)


# for e in event_tests for g in given_tests)


def entropy(prob: float, base=2) -> float:
	value = 0
	if prob != 0 and base not in {0, 1}:
		value = - math.log(prob, base) * prob
	return value


# todo: get it to work using two tests instead of one
def probability(
		event: Collection,
		event_test: Callable = None,
		given: Collection = None,
		given_test: Callable = None) -> Union[float, Mapping[Any, float]]:
	pr = collections.defaultdict(float)
	if event_test is None:
		counts = collections.Counter(event)
		pr.update({e: freq / len(event) for e, freq in counts.items()})
	else:
		pr = sum(map(event_test, event)) / len(event)
	if given is not None:
		min_len = min(len(event), len(given))
		g_counts = collections.Counter(given)
		pr_g = collections.defaultdict(float)
		pr_g.update({g: freq / len(given) for g, freq in g_counts.items()})
		joint_events = [(e, g) for e, g in zip(event, given)]
		joint_counts = collections.Counter(joint_events)
		if given_test is None:
			pr.clear()
			if event_test is None:
				pr.update({
					eg: (freq / min_len) / pr_g[eg[1]]
					for eg, freq in joint_counts.items()})
			else:
				pr.update({
					eg: (freq / min_len) / pr_g[eg[1]]
					for eg, freq in joint_counts.items() if event_test(eg[0])})
		else:  # todo add flag to see if we want 'not' vs 'yes'.
			if event_test is None:
				pr.clear()
				pr.update({
					eg: (freq / min_len) / pr_g[eg[1]]
					for eg, freq in joint_counts.items() if given_test(eg[1])})
			else:
				eg_freq = sum(
					1 for e, g in joint_events
					if event_test(e) and given_test(g))
				# Pr(Y, X) / Pr(X) = Pr(Y|X)
				pr = (eg_freq / min_len) / pr if pr != 0 else 0
	return pr
