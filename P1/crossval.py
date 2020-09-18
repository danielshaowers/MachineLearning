'https://medium.com/dev-genius/python-for-experienced-programmers-a2ee334ce62f'
# let's use dictionaries for our data
import random
from typing import Any, Iterable


def find_indices(list, condition):
	return [i for i, elem in enumerate(list) if condition(elem)]


def cross_val(labels: Iterable[Any], folds: int):
	pos_lab = find_indices(labels, lambda x: x > 0)
	neg_lab = find_indices(labels, lambda x: x <= 0)
	pos_per_fold, pos_remainder = divmod(len(pos_lab), folds)
	neg_per_fold, neg_remainder = divmod(len(neg_lab), folds)

	random.shuffle(pos_lab)
	random.shuffle(neg_lab)
	data = []
	for x in range(folds):
		pos_count = pos_per_fold + min(1, pos_remainder)
		neg_count = neg_per_fold + min(1, neg_remainder)
		data[x] = [pos_lab[:pos_count] + neg_lab[:neg_count]]
		pos_remainder = max(0, pos_remainder - 1)
		neg_remainder = max(0, neg_remainder - 1)
		del pos_lab[:pos_count]
		del neg_lab[:neg_count]
	return data
