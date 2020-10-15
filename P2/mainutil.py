import argparse
import itertools
import statistics

import crossval
import mldata
import mlutil
import model


def p2_main(
		path: str,
		learner: model.Model,
		skip_cv: bool,
		print_results: bool = True,
		save_as: str = None):
	file_base, root_dir = get_dataset_and_path(path)
	data = mldata.parse_c45(file_base=file_base, rootdir=root_dir)
	n_folds = 1 if skip_cv else 5
	predictions, labels = crossval.cross_validate(
		learner=learner, data=data, n_folds=n_folds, save_as=save_as
	)
	accuracies = []
	precisions = []
	recalls = []
	for predicted, truths in zip(predictions, labels):
		accuracy, precision, recall, _, _ = mlutil.prediction_stats(
			scores=predicted, truths=truths, threshold=0.5
		)
		accuracies.append(accuracy)
		precisions.append(precision)
		recalls.append(recall)
	results = {
		'mean_accuracy': statistics.mean(accuracies),
		'mean_precision': statistics.mean(precisions),
		'mean_recall': statistics.mean(recalls),
	}
	sd_accuracy = statistics.stdev(accuracies) if n_folds > 1 else 0
	sd_precision = statistics.stdev(precisions) if n_folds > 1 else 0
	sd_recall = statistics.stdev(recalls) if n_folds > 1 else 0
	results.update({
		'sd_accuracy': sd_accuracy,
		'sd_precision': sd_precision,
		'sd_recall': sd_recall,
	})
	all_preds = tuple(itertools.chain.from_iterable(predictions))
	all_labels = tuple(itertools.chain.from_iterable(labels))
	auc, best_thresh = mlutil.compute_roc(scores=all_preds, truths=all_labels)
	results.update({
		'auc': auc,
		'best_threshold': best_thresh
	})
	if print_results:
		print_p2_results(
			mean_accuracy=round(results['mean_accuracy'], 4),
			sd_accuracy=round(results['sd_accuracy'], 4),
			mean_precision=round(results['mean_precision'], 4),
			sd_precision=round(results['sd_precision'], 4),
			mean_recall=round(results['mean_recall'], 4),
			sd_recall=round(results['sd_recall'], 4),
			mean_roc=round(auc, 4),
			best_roc_threshold=round(best_thresh, 4)
		)
	return results


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
