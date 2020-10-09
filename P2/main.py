import random
from sys import argv

import numpy as np

import mldata
import mlutil


def mainm(dataset, data_path, use_cv, max_depth, use_info_gain: int):
    # only relevant for when we're running the experiment
    data = mldata.parse_c45(dataset, data_path)
    npdata = mlutil.convert_to_numpy(data)
    labels = np.array(mlutil.get_labels(data))
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
    mainm(dataset, data_path, use_cv, max_depth, use_info_gain)
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
    dataset = 'voting'
    use_cv = True
    max_depth = 1
    use_info_gain = 0
    max_depth = int(max_depth)
    use_cv = bool(use_cv)
    use_info_gain = bool(use_info_gain)
    mainm(dataset, data_path, use_cv, max_depth, use_info_gain)
