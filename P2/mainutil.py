import argparse
import itertools
import statistics

import crossval
import mldata
import mlutil
import model


def p2_main(path: str, learner: model.Model, skip_cv: bool):
	file_base, root_dir = get_dataset_and_path(path)
	data = mldata.parse_c45(file_base=file_base, rootdir=root_dir)
	n_folds = 1 if skip_cv else 5
	predictions, labels = crossval.cross_validate(
		model=learner, data=data, n_folds=n_folds)
	accuracies = []
	precisions = []
	recalls = []
	for predicted, truths in zip(predictions, labels):
		accuracy, precision, recall, _ = mlutil.prediction_stats(
			scores=predicted, truths=truths, threshold=0.5
		)
		accuracies.append(accuracy)
		precisions.append(precision)
		recalls.append(recall)
	results = {
		'mean_accuracy': round(statistics.mean(accuracies), 4),
		'sd_accuracy': round(statistics.stdev(accuracies), 4),
		'mean_precision': round(statistics.mean(precisions), 4),
		'sd_precision': round(statistics.stdev(precisions), 4),
		'mean_recall': round(statistics.mean(recalls), 4),
		'sd_recall': round(statistics.stdev(recalls), 4),
	}
	all_preds = tuple(itertools.chain.from_iterable(predictions))
	all_labels = tuple(itertools.chain.from_iterable(labels))
	auc, best_thresh = mlutil.compute_roc(scores=all_preds, truths=all_labels)
	print_p2_results(
		mean_accuracy=results['mean_accuracy'],
		sd_accuracy=results['sd_accuracy'],
		mean_precision=results['mean_precision'],
		sd_precision=results['sd_precision'],
		mean_recall=results['mean_recall'],
		sd_recall=results['sd_recall'],
		mean_roc=round(auc, 4),
		best_roc_threshold=round(best_thresh, 4)
	)


def base_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--path',
		type=str,
		help='The path to the data',
		required=True
	)
	parser.add_argument(
		'--skip_cv',
		type=int,
		help='1 to skip cross validation; 0 to use cross validation',
		required=True
	)
	return parser


def print_p2_results(
		mean_accuracy: float,
		sd_accuracy: float,
		mean_precision: float,
		sd_precision: float,
		mean_recall: float,
		sd_recall: float,
		mean_roc: float,
		best_roc_threshold: float):
	print(f'Accuracy: {mean_accuracy} {sd_accuracy}')
	print(f'Precision: {mean_precision} {sd_precision}')
	print(f'Recall: {mean_recall} {sd_recall}')
	print(f'Area under ROC: {mean_roc}')
	print(f'Best threshold: {best_roc_threshold}')


def get_dataset_and_path(path: str):
	split_path = path.split('\\')
	data_set = split_path[-1]
	data_path = '\\'.join(split_path[:-1])
	return data_set, data_path
