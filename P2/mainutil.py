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
		is_experiment: bool = False,
		save_as: str = None):
	file_base, root_dir = get_dataset_and_path(path)
	data = mldata.parse_c45(file_base=file_base, rootdir=root_dir)
	n_folds = 1 if skip_cv else 5
	predictions, labels = crossval.cross_validate(
		learner=learner, data=data, n_folds=n_folds, save_as=save_as
	)
	adjustments = None
	if is_experiment:
		adjustments = [n for _, n in predictions]
		predictions = [p for p, _ in predictions]
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
	if not is_experiment:
		auc, best_thresh = mlutil.compute_roc(
			scores=all_preds, truths=all_labels
		)
		results.update({
			'auc': auc,
			'best_threshold': best_thresh
		})
	if is_experiment:
		results.update({
			'mean_adjustments': statistics.mean(adjustments),
			'sd_adjustments': statistics.stdev(adjustments)
			if n_folds > 1 else 0
		})
	if print_results:
		if is_experiment:
			print_p2_results(
				mean_accuracy=round(results['mean_accuracy'], 4),
				sd_accuracy=round(results['sd_accuracy'], 4),
				mean_precision=round(results['mean_precision'], 4),
				sd_precision=round(results['sd_precision'], 4),
				mean_recall=round(results['mean_recall'], 4),
				sd_recall=round(results['sd_recall'], 4),
				mean_adjustments= round(results['mean_adjustments'], 4),
				sd_adjustments=round(results['sd_adjustments'], 4)
			)
		else:
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
		mean_accuracy: float = None,
		sd_accuracy: float = None,
		mean_precision: float = None,
		sd_precision: float = None,
		mean_recall: float = None,
		sd_recall: float = None,
		mean_roc: float = None,
		best_roc_threshold: float = None,
		mean_adjustments: float = None,
		sd_adjustments: float = None):
	if mean_accuracy is not None and sd_accuracy is not None:
		print(f'Accuracy: {mean_accuracy} {sd_accuracy}')
	if mean_precision is not None and sd_precision is not None:
		print(f'Precision: {mean_precision} {sd_precision}')
	if mean_recall is not None and sd_recall is not None:
		print(f'Recall: {mean_recall} {sd_recall}')
	if mean_roc is not None:
		print(f'Area under ROC: {mean_roc}')
	if best_roc_threshold is not None:
		print(f'Best threshold: {best_roc_threshold}')
	if mean_adjustments is not None and sd_adjustments is not None:
		print(f'Adjustments: {mean_adjustments} {sd_adjustments}')



def get_dataset_and_path(path: str):
	split_path = path.split('\\')
	data_set = split_path[-1]
	data_path = '\\'.join(split_path[:-1])
	return data_set, data_path
