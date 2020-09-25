from typing import Tuple

from .mldata import ExampleSet, Feature


def get_features(example_set: ExampleSet, example_index: int) -> Tuple:
	return tuple(example_set[example_index][:-1])


def get_label(example_set: ExampleSet, example_index: int):
	return example_set[example_index][-1]


def get_feature_info(example_set: ExampleSet, feature_index: int) -> Feature:
	return example_set.schema[feature_index]
