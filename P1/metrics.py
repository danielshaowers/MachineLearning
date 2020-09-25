import random
from collections import Counter
from typing import Any, Iterable, SupportsFloat, SupportsInt, Union
import statistics
import numpy as np
from collections import deque
from P1 import algorithm
import copy
Number = Union[SupportsInt, SupportsFloat]


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
