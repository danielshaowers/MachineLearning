from typing import Iterable

from observation import ObservationSet
from mldata import Feature


# TODO Develop data structure for tree: linked list?

def id3(dataset: ObservationSet, target: Feature, features: Iterable[Feature]):
	"""

	Args:
		dataset: Collection of training examples.
		target: Feature whose value is to be predicted by the tree.
		features: Collection of other features that may be tested by the
			learned decision tree.

	Returns:
		A decision tree that classifies the given observations.
	"""

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
