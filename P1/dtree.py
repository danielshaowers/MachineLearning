# from src import algorithm, crossval
import statistics
from sys import argv

import algorithm
import crossval
import metrics
import mldata


# path: str, skip_cv: bool, max_depth: int, use_info_gain: bool
def main() -> None:
	temp, path, skip_cv, max_depth, use_info_gain = argv
	print('accepted arguments ' + path + skip_cv + max_depth + use_info_gain)


def main(path: str, skip_cv: bool, max_depth: int, use_info_gain: bool) -> None:
	# in the real thing there will be a command line input and we need to
	# parse the arguments that way
	"""
	Args:
		path: Path to the data. If this is “/a/b/someproblem” then you
			will load “/a/b/someproblem.names” and “/a/b/someproblem.data”.
		skip_cv: Skip cross validation if true; otherwise, build the model
			using the entire data set.
		max_depth: A non-negative integer that sets the maximum depth of the
			tree (the number of tests on any path from root to leaf). If this
			value is zero, you should grow the full tree. If this value is
			positive, grow the tree to the given value. Note that if this
			value is too large, it will have the same effect as when the
			option is zero.
		use_info_gain: If false, use information gain for the split
		criterion; otherwise, use gain ratio.

	Returns:
		None

	Raises:
	"""

	data = mldata.parse_c45(path)

	if use_info_gain:
		split_criteria = metrics.info_gain
	else:
		split_criteria = metrics.gain_ratio
	learner = algorithm.ID3(max_depth=max_depth, split_function=split_criteria)

	# experiment: multiple iterations and use the majority label
	accuracy = []
	if skip_cv == 0:
		foldNum = 5
		folds = crossval.get_folds(data, foldNum)
		for i in range(foldNum):
			train, test = crossval.get_train_test_split(folds, i)
			accuracy[i], plabels = train_test(learner, train, test)
		accuracy = statistics.mean(accuracy)
	else:
		accuracy, plabels = train_test(data, data, use_info_gain, max_depth)


# research idea: at each step of the decision tree, instead of connsidering
# the full dataset, create n subsets. for each of these subsets, add on a
# random, balanced number of examples from other subsets.
# find the feature partition that has the maximum average information gain
# this addresses the variance withinn decision trees by considering the
# feature most resistant to variance while  still ensuring that all the data
# is used at least once
def train_test(train, test, use_info_gain, max_depth):
	train_labels = set(o.label for o in train)
	test_labels = set(o.label for o in test)
	dtree = algorithm.id3(train, train_labels, use_info_gain, max_depth)
	plabels = algorithm.predict(dtree, test)


def train_test(learner: algorithm.Model, train_set, test_set):
	train_labels = set(o.label for o in train_set)
	test_labels = set(o.label for o in test_set)
	dtree = learner.train(train_set)
	plabels = learner.predict(dtree, train_set)
	match = 0
	plab = enumerate(plabels)
	for idx, lab in plab:
		if lab == test_labels[idx]:
			match = match + 1
	accuracy = match / len(plabels)
	return plabels, accuracy


if __name__ == "__main__":
	main()
