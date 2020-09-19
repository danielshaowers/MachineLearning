from typing import Iterable

from observation import ObservationSet
from mldata import Feature
from collections import deque
import numpy
# TODO Develop data structure for tree: linked list?
import P1.metrics
"""

	Args:
		dataset: Collection of training examples.
		target: Feature whose value is to be predicted by the tree.
		features: Collection of other features that may be tested by the
			learned decision tree.

	Returns:
		A decision tree that classifies the given observations.
	"""

def id3(dataset: ObservationSet, target: Feature, features: Iterable[Feature]):

	pure_node = checkPure(target.to_float)
	for feat in features:
			if feat.Type == Feature.Type.NOMINAL:
				ig = P1.gain_ratio
			elif feat.Type == Feature.Type.CONTINUOUS: #consider every boolean test
					partition = get_best_partition(feat)

	#root = Node(bestF)a


def checkPure(iterator):
   return len(set(iterator)) <= 1

def get_best_partition(feat, labels): #find every time the class label changes
	splits = deque()

	labelsN = numpy.array(labels)
	featN=numpy.array(feat)
	sorted_indices = numpy.argsort(featN)
	featN=numpy.sort(featN)
	labelsN = labelsN[sorted_indices]
	prev_type = labelsN[0]
	splits.append(0)
	for x in range(1, len(labelsN)):
		if labelsN[x] != prev_type:
			#splits.append(featN(x)) #add to split if the type changes so we don't check every single label
			splits.append(x)
		prev_type = labelsN[x]
	bestSplit = 0
	while(splits): #could make this a dp problem. or greedy maybe? while splits is not empty - it'd be faster to sort feature first and get labels to correspond.
		finish = splits.pop()
		pos_belowCount = sum(labelsN[1:finish])
		neg_belowCount = finish - pos_belowCount
		pos_aboveCount = sum(labelsN[finish:len(labelsN)])
		neg_aboveCount = len(labelsN) - pos_aboveCount
		return bestSplit
def find_indices(list, condition):
	return [i for i, elem in enumerate(list) if condition(elem)]

"""
	ID3 pseudo-code (from Machine Learning textbook)
	------------------------------------------------
	1. Create a root node for the tree
	2. If all observations in the dataset are of the same class, return the 
		single-node tree with that class label
	3. If features is empty, return the single-node tree with the target 
		label that is most common among the observations
	4. Otherwise:
		4.1 Let F be the feature in features that "best" classifies 
			observations
		4.2 Let root be F
		4.3 For each possible value f_i of F:
			4.3.1 Add a new tree branch below root, corresponding to the test
				F = f_i (this needs to be generalized to account for 
				continuous features as well)
			4.3.2 Let features_fi be the subset of features that have value 
				f_i for F
			4.3.3 If features_fi is empty, then below this new branch, 
				add a leaf node with the most common value of target among the 
				observations as the label
			4.3.4 Otherwise, below this new branch, add the subtree
				id3(features_fi, target, features - {F})
	Return root
	"""