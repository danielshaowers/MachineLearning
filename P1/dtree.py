from sys import argv
from typing import NoReturn

import algorithm
import crossval
import metrics
import mldata
import mlutil


def main() -> NoReturn:

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
	"""
	try:
		temp, path, skip_cv, max_depth, use_info_gain = argv
		print('accepted arguments ' + path + skip_cv + max_depth + use_info_gain)
	except:
		path = 'spam'
		skip_cv = 0
		max_depth = 10
		use_info_gain = 1
	max_depth = int(max_depth)
	skip_cv = bool(skip_cv)
	use_info_gain = bool(use_info_gain)
	# def main(path: str, skip_cv: bool, max_depth: int, use_info_gain: bool)
	# -> None:
	# in the real thing there will be a command line input and we need to
	# parse the arguments that way
	#todo:
	data = mldata.parse_c45(path, "C:/users/danie/PycharmProjects/")  # this
	# input might have to be fixed? i couldn't get it work as an absolute path
	all_labels = mlutil.get_labels(data)

	split_criteria = metrics.info_gain if use_info_gain else metrics.gain_ratio

	# initialize the tests, but don't do any training yet
	learner = algorithm.ID3(max_depth=max_depth, split_function=split_criteria)
	# experiment: multiple iterations and use the majority label
	accuracy = []
	tree_size = []
	first_feat = []
	depth = []
	if skip_cv:
		fold_num = 5
		folds = crossval.get_folds(data, fold_num)
		for i in range(fold_num):
			train, test = crossval.get_train_test_split(folds, i)
			acc, pred, size, root_feat, d = train_test(learner, train, test)
			accuracy.append(acc)
			tree_size.append(size)
			first_feat.append(root_feat)
			depth.append(d)
			print(f'Results of {i}th fold')
			print('-----------------------')
			print(f'Accuracy: {acc[i]}')
			print(f'Size: {tree_size[i]}')
			print(f'Maximum Depth: {depth[i]}')
			print(f'First Feature: {root_feat[i]}')
	else:
		acc, pred, acc, tree_size, depth = train_test(learner, data, data)
		print('Average Metrics')
		print('-----------------------')
		print(f'Accuracy: {sum(acc) / len(acc)}')
		print(f'Size: {sum(tree_size) / len(tree_size)}')
		print(f'Maximum Depth: {sum(depth) / len(depth)}')


"""
research idea: at each step of the decision tree, instead of considering
the full dataset, create n subsets. for each of these subsets, add on a
random, balanced number of examples from other subsets.
find the feature partition that has the maximum average information gain
this addresses the variance within decision trees by considering the
feature most resistant to variance while  still ensuring that all the data
is used at least once
"""


# TODO What are train_set and test_set?
def train_test(learner: algorithm.ID3, train_set, test_set):
	# TODO Not using this?
	train_labels = mlutil.get_labels(train_set)
	test_labels = mlutil.get_labels(test_set)
	learner.train(train_set)
	# TODO Shouldn't this be test_test?
	pred_labels = learner.predict(train_set)
	n_correct = sum(1 for p, t in zip(pred_labels, test_labels) if p == t)
	accuracy = n_correct / len(pred_labels)
	size = learner.model_metrics[algorithm.ID3.Metrics.TREE_SIZE]
	depth = learner.model_metrics[algorithm.ID3.Metrics.MAX_DEPTH]
	first_feat = learner.model_metrics[algorithm.ID3.Metrics.FIRST_FEATURE]
	return pred_labels, accuracy, size, first_feat, depth


if __name__ == "__main__":
	main()
