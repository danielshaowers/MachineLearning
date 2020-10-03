import itertools
import random
import statistics
from sys import argv

import algorithm
import crossval
import metrics
import mldata
import mlutil


def print_results(acc, size, depth, first_feat=None):
    print(f'Accuracy: {acc}')
    print(f'Size: {size}')
    print(f'Maximum Depth: {depth}')
    if first_feat is not None:
        print(f'First Feature: {first_feat}')


def train_test(
        learner: algorithm.ID3,
        train_set: mldata.ExampleSet,
        test_set: mldata.ExampleSet):
    test_labels = mlutil.get_labels(test_set)
    learner.train(train_set)
    pred_labels = learner.predict(test_set)
    n_correct = sum(
        itertools.starmap(lambda p, t: p == t, zip(pred_labels, test_labels)))
    pred_labels = learner.predict(test_set)
    accuracy = n_correct / len(pred_labels)
    size = learner.model_metrics[algorithm.ID3.Metrics.TREE_SIZE]
    depth = learner.model_metrics[algorithm.ID3.Metrics.MAX_DEPTH]
    first_node = learner.model_metrics[algorithm.ID3.Metrics.FIRST_NODE]
    return pred_labels, accuracy, size, first_node, depth


def main(dataset, data_path, use_cv, max_depth, use_info_gain: int):
    # only relevant for when we're running the experiment
    partition_count = [3, 5, 7, 10] if use_info_gain < 0 else [1]
    data = mldata.parse_c45(dataset, data_path)
    data = mldata.ExampleSet([e for i,e in enumerate(data) if i < 1000]) 
    if use_info_gain >= 1:
        split_criteria = metrics.info_gain
    elif use_info_gain == 0:
        split_criteria = metrics.gain_ratio
    else:
        split_criteria = metrics.stochastic_information_gain

    for z in partition_count:  # run the experiment
        if len(partition_count) > 1:
            print(f'\nrunning experiment with {z} partitions')
        learner = algorithm.ID3(
            max_depth=max_depth, split_function=split_criteria, partitions=z)
        run(use_cv, data, learner)


def run(use_cv, data, learner):
    accuracies = []
    tree_sizes = []
    roots = []
    depths = []
    if use_cv == 0:
        fold_num = 5
        folds = crossval.get_folds(data, fold_num)
        for i in range(fold_num):
            train, test = crossval.get_train_test_split(folds, i)
            p, acc, size, first_node, depth = train_test(learner, train, test)
            accuracies.append(acc)
            tree_sizes.append(size)
            roots.append(first_node.feature)
            depths.append(depth)
            print(f'\n Results of {i + 1}th fold')
            print('-----------------------')
            print(accuracies[i], tree_sizes[i], depths[i], roots[i].feature)
            print(
                f'positive guesses / positive truth labels'
                f' {sum(p) / sum(mlutil.get_labels(train))}')
            print(p)
        print('\n Average Metrics')
        print('-----------------------')
        print_results(
            statistics.mean(accuracies),
            statistics.mean(tree_sizes),
            statistics.mean(depths))
        return accuracies, tree_sizes, roots, depths
    else:
        _, acc, tree_sizes, first_node, depth = train_test(learner, data, data)
        print('Average Metrics')
        print('-----------------------')
        print_results(acc, tree_sizes, depth, first_node)
        return acc, tree_sizes, first_node, depth
    return accuracies
if __name__ == "__main__":
    random.seed(a=12345)
    try:
        _, path, use_cv, max_depth, use_info_gain = argv
        max_depth = int(max_depth)
        use_cv = bool(use_cv)
        use_info_gain = bool(use_info_gain)
        split_path = path.split('\\')
        dataset = split_path[-1]
        data_path = '\\'.join(split_path[:-1])
        print('Accepted Arguments')
        print('-----------------------')
        print(f'Path: {path}')
        print(f'Use cross validation: {use_cv}')
        print(f'Maximum tree depth: {max_depth}')
        print(f'Use information gain: {use_info_gain}')
        main(dataset, data_path, use_cv, max_depth, use_info_gain)
    except ValueError:
        print('Not all arguments were provided. Provide the following:')
        print('dtree.py <path> <skip_cv> <max_depth> <use_info_gain>')
        print('<path> \t\t\t\t Path to the data\n')
        print('<skip_cv> \t\t\t Skip cross validation if 1\n')
        print('<max_depth> \t\t Non-negative integer that sets the maximum')
        print('\t\t\t\t\t depth of the tree. If value is zero, you should')
        print('\t\t\t\t\t grow the full tree\n')
        print('<use_info_gain> \t Use information gain as the split')
        print('\t\t\t\t\t criteria. Otherwise, use gain ratio')
        data_path = '..'
        dataset = 'spam'
        use_cv = True
        max_depth = 1
        use_info_gain = 0
        max_depth = int(max_depth)
        use_cv = bool(use_cv)
        use_info_gain = bool(use_info_gain)
        main(dataset, data_path, use_cv, max_depth, use_info_gain)

