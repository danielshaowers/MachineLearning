from P1.observation import ObservationSet
from P1.mldata import Feature
from collections import deque
import numpy

import P1.metrics
from P1 import Node

"""

	Args:
		dataset: Collection of training examples.
		target: Feature whose value is to be predicted by the tree.
		features: Collection of other features that may be tested by the
			learned decision tree.

	Returns:
		A decision tree that classifies the given observations.
	"""
#for continuous and nominal, we don't need to remove them after we come up with one test. we just remove if it's a pure node
# or track how many tests exist for nominal and continuous
#ask for clarification what these each are. what is features vs dataset? is dataset just a compilation of label and features?

def id3(dataset: ObservationSet, target: Feature, parent_node: Node, igflag):
    if checkpure(target.to_float()) | len(features) == 0:
		return parent_node #not sure what it should return? maybe true or false given left or right child

    for feat in features: #find the best feature
        if feat.Type == Feature.Type.NOMINAL:
			if igflag:
				ig = P1.information_gain()
			else:
            	ig = P1.gain_ratio()
		else:
            partition = get_best_partition(feat, target, igflag)



	feat_array = numpy.array(best_feat)
	lChild_idx = numpy.argwhere(feat_array <= partition)
	rChild_idx = numpy.argwhere(feat_array > partition)
	leftChild = parent_node.left(Node(parent_node, partition))
	rightChild = parent_node.right(Node(parent_node, partition))


# root = Node(bestF)a


def checkpure(iterator):  # boolean if all labels are in the same class
    return len(set(iterator)) <= 1

#criterion = 1 for information gain, 0 for gain ratio
def get_best_partition(feat, labels, use_ig):  # find every time the class label changes
#	both = [(example, label)];
#	sort(both, key=lambda k:k[0]) #this sorts by col 1
    splits = deque()
    labelsN = numpy.array(labels)
    featN = numpy.array(feat)
    sorted_indices = numpy.argsort(featN)
    featN = numpy.sort(featN)
    labelsN = labelsN[sorted_indices] #sort according to feature values
    prev_type = labelsN[0]
    splits.append(0)
    for x in range(1, len(labelsN)):
        if labelsN[x] != prev_type:
            # splits.append(featN(x)) #add to split if the type changes so we don't check every single label
            splits.append(x)
        prev_type = labelsN[x]
    bestSplit = 0
	scoring = numpy.array(splits)
	cnt = 0;
	while splits:  # could make this a dp problem. or greedy maybe? while splits is not empty - it'd be faster to sort
		# feature first and get labels to correspond.
		finish = splits.pop()
		split_feats = numpy.zeros(finish) + numpy.zeros(len(labelsN) - finish)
		if use_ig:
			scoring[cnt] = P1.information_gain(split_feats, labelsN) #feed in the labels and
		else:
			scoring[cnt] = P1.gain_ratio(split_feats, labelsN)
		cnt=cnt+1
	return featN[numpy.argmax(scoring)] #this returns the best split to use. however...it doesn't show directionality



def find_indices(list, condition):
    return [i for i, elem in enumerate(list) if condition(elem)]

#return predicted labels
def predict(test_data: ObservationSet):

"""
	ID3 pseudo-code (from Machine Learning textbook)
	------------------------------------------------d
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
