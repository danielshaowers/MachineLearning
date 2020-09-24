from P1 import crossval, algorithm
import statistics
import argparse
def main(path: str, skip_cv: bool, max_depth: int, use_info_gain: bool) -> None:
	#in the real thing there will be a command line input and we need to parse the arguments that way
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
	#experiment: multiple iterations and use the majority label
	with open(path, 'r') as f:
		data= f.read
	accuracy = []
	if skip_cv == 0:
		foldNum = 5
		folds = crossval.getfolds(data, foldNum)
		for i in range(foldNum):
			train, test = crossval.get_train_test_split(folds, i)
			accuracy[i], plabels = train_test(train, test, use_info_gain, max_depth)
		accuracy = statistics.mean(accuracy)
	else:
		accuracy, plabels = train_test(data, data, use_info_gain, max_depth)



def train_test(train, test, use_info_gain, max_depth)
	train_labels = set(o.label for o in train)
	test_labels = set(o.label for o in test)
	dtree = algorithm.id3(train, train_labels, use_info_gain, max_depth)
	plabels = algorithm.predict(dtree, test)
	match = 0
	plab = enumerate(plabels)
	for idx, lab in plab:
		if lab == test_labels[idx]:
			match = match + 1
	accuracy = match / len(plabels)
	return plabels, accuracy