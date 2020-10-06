import collections
import copy
import functools
import math
import random
import statistics
from typing import Any, Callable, Collection, Mapping
from typing import Iterable, Union

"""
Research idea: at each step of the decision tree, instead of considering
the full dataset, create n subsets. for each of these subsets, add on a
random, balanced number of examples from other subsets.
find the feature partition that has the maximum average information gain
this addresses the variance within decision trees by considering the
feature most resistant to variance while  still ensuring that all the data
is used at least once
"""


# shuffle the training data into n blocks with a random number of additional
# units from other blocks
def shuffle_blocks(labels: Iterable, blocks):
	pos_lab = find_indices(labels, lambda y: y > 0)
	neg_lab = find_indices(labels, lambda y: y <= 0)
	pos_per_fold, pos_remainder = divmod(len(pos_lab), blocks)
	neg_per_fold, neg_remainder = divmod(len(neg_lab), blocks)

	random.shuffle(pos_lab)
	random.shuffle(neg_lab)
	data = [None] * blocks
	pos_lab_copy = copy.deepcopy(pos_lab)
	neg_lab_copy = copy.deepcopy(neg_lab)
	# how many additional examples we want not in the block per pos lab and
	# neg lab
	additional_pos = round(pos_per_fold * blocks / 2)
	additional_neg = round(neg_per_fold * blocks / 2)

	for y in range(blocks):
		pos_count = pos_per_fold + min(1, pos_remainder)
		neg_count = neg_per_fold + min(1, neg_remainder)
		pos_remainder = max(0, pos_remainder - 1)
		neg_remainder = max(0, neg_remainder - 1)
		data[y] = pos_lab[:pos_count] + neg_lab[:neg_count]
		del pos_lab[:pos_count]
		del neg_lab[:neg_count]
		# get pool of positives and pool of negatives that aren't in data[x]
		pos_pool = [x for x in pos_lab_copy if x not in data[y]]
		neg_pool = [x for x in neg_lab_copy if x not in data[y]]
		pos_deck = list(range(len(pos_pool)))
		neg_deck = list(range(len(neg_pool)))
		random.shuffle(pos_deck)
		random.shuffle(neg_deck)
		data[y] = data[y] + pos_deck[:additional_pos] + neg_deck[:additional_neg]
	return data  # indices indicating which for each block


# def stochastic_information_gain(features, labels, partitions, use_ig):
def stochastic_information_gain(
		event: Collection,
		event_tests: Collection[Callable],
		given: Collection,
		given_tests: Collection[Callable],
		partitions: int = 5) -> float:
	listed_indices = shuffle_blocks(event, partitions)
	ig = []
	for i in range(len(listed_indices)):
		subset_feats = [given[g] for g in listed_indices[i]]
		subset_labels = [event[g] for g in listed_indices[i]]
		# if use_ig:
		# 	ig[i] = info_gain(subset_labels, subset_feats)
		# else:
		try:
			ig.append(
				gain_ratio(
					subset_labels, event_tests, subset_feats, given_tests, 1))
		except:
			given
	return statistics.mean(ig)


def find_indices(list, condition):
	return [i for i, elem in enumerate(list) if condition(elem)]


def gain_ratio(
		event: Collection,
		event_tests: Collection[Callable],
		given: Collection,
		given_tests: Collection[Callable]) -> float:
	"""Computes the gain ratio over an event and given random variables.

	Tests that evaluate event values and given values are used to compute the
	conditional and unconditional probabilities used when computing the
	information gain.

	For a continuous given variable, it is convention to only provide a single
	given test, which is testing if values of the given variable are greater
	than some threshold. For binary and continuous variables, tests that
	evaluated all possible values of the variable are expected as input.

	Args:
		event: Collection of values representing the event random variable.
		event_tests: Tests on which to evaluate each value of event.
		given: Collection of values representing the given random variable.
		given_tests: tests on which to evaluate each value of given.

	Returns:
		Gain ratio of the event and a given random variable.
	"""
	event_entropy = sum(entropy(probability(event, e)) for e in event_tests)
	return info_gain(event, event_tests, given, given_tests) / event_entropy


def info_gain(
		event: Collection,
		event_tests: Collection[Callable],
		given: Collection,
		given_tests: Collection[Callable]) -> float:
	"""Computes the information gain over an event and given random variables.

	Tests that evaluate event values and given values are used to compute the
	conditional and unconditional probabilities used when computing the
	information gain.

	For a continuous given variable, it is convention to only provide a
	single given test, which is testing if values of the given variable are
	greater than some threshold. For binary and continuous variables,
	tests that evaluated all possible values of the variable are expected as
	input.

	Args:
		event: Collection of values representing the event random variable.
		event_tests: Tests on which to evaluate each value of event.
		given: Collection of values representing the given random variable.
		given_tests: tests on which to evaluate each value of given.

	Returns:
		Information gain of the event and a given random variable.
	"""
	cond_entropy = conditional_entropy(event, event_tests, given, given_tests)
	event_entropy = sum(entropy(probability(event, e)) for e in event_tests)
	gain = event_entropy - cond_entropy
	return 0 if math.isnan(gain) else gain


def conditional_entropy(
		event: Collection,
		event_tests: Collection[Callable],
		given: Collection,
		given_tests: Collection[Callable]) -> float:
	"""Computes the conditional entropy over an event and given random
	variables.

	Tests that evaluate event values and given values are used to compute the
	conditional and unconditional probabilities used when computing the
	conditional entropy.

	For a continuous given variable, it is convention to only provide a
	single given test, which is testing if values of the given variable are
	greater than some threshold. For binary and continuous variables,
	tests that evaluated all possible values of the variable are expected as
	input.

	Args:
		event: Collection of values representing the event random variable.
		event_tests: Tests on which to evaluate each value of event.
		given: Collection of values representing the given random variable.
		given_tests: tests on which to evaluate each value of given.

	Returns:
		Conditional entropy of the event and a given random variable.
	"""
	gvn_tests = given_tests
	if len(given_tests) == 1:
		new_boi = [functools.partial(lambda x: not g(x)) for g in given_tests]
		gvn_tests = [*given_tests, *new_boi]
	return sum(
		probability(given, g) * entropy(probability(event, e, given, g))
		for e in event_tests for g in gvn_tests)


def entropy(prob: float, base=2) -> float:
	"""Computes the entropy of a probability.
	Args:
		prob: Probability to use when computing entropy.
		base: Base of the logarithm function.

	Returns:
		Shannon entropy.
	"""
	value = 0
	if prob != 0 and base not in {0, 1}:
		value = - math.log(prob, base) * prob
	return value


def probability(
		event: Collection,
		event_test: Callable[[Any], bool] = None,
		given: Collection = None,
		given_test: Callable[[Any], bool] = None,
		m: float = 0,
		p: float = 0) -> Union[float, Mapping[Any, float]]:
	"""Computes either the unconditional or conditional probability.

	If only the event is specified, the probability of each value it takes on
	is computed. If the event test is specified then only the probability
	that the event variable satisfies that test will be computed.

	A similar procedure is followed for all possible combinations of event,
	event test, given, and given test.

	m-estimates can be applied to the conditional probability calculation.
	The default values of m and p are 0 so the empirical estimate is
	computed. To apply Laplace smoothing, let v be the number of distinct
	values the event takes on. Then let m = v and p = 1 / v.

	Args:
		event: Collection of values representing the event random variable.
		event_test: Test on which to evaluate each value of event.
		given: Collection of values representing the given random variable.
		given_test: Test on which to evaluate each value of given.
		m: "Equivalent sample size" which determines the important of p
			relative to the observations.
		p: Prior estimate of the probability.

	Returns:
		A dictionary of probabilities or a single float probability.
	"""

	def conditional(joint_freq, given_freq):
		return (joint_freq + m * p) / (given_freq + m)

	# Default probability of something not in the dictionary is 0.0.
	pr = collections.defaultdict(float)
	if event_test is None:
		counts = collections.Counter(event)
		pr.update({e: freq / len(event) for e, freq in counts.items()})
	else:
		pr = sum(map(event_test, event)) / len(event)
	if given is not None:
		g_counts = collections.Counter(given)
		joint = [(e, g) for e, g in zip(event, given)]
		joint_counts = collections.Counter(joint)
		pr.clear()
		if given_test is None:
			if event_test is None:
				pr.update({
					eg: conditional(freq, g_counts[eg[1]])
					for eg, freq in joint_counts.items()})
			else:
				pr.update({
					eg: conditional(freq, g_counts[eg[1]])
					for eg, freq in joint_counts.items() if event_test(eg[0])})
		else:
			if event_test is None:
				pr.update({
					eg: conditional(freq, g_counts[eg[1]])
					for eg, freq in joint_counts.items() if given_test(eg[1])})
			else:
				eg_freq = sum(
					1 for e, g in joint if event_test(e) and given_test(g))
				pr = conditional(eg_freq, min(len(event), len(given)))
	return pr
