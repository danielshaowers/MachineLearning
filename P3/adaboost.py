import json
import random
from copy import copy
from typing import NoReturn

import numpy as np

import dtree
import logreg
import nbayes
import P1.algorithm
import metrics
import mainutil
from mldata import ExampleSet
import mldata
import model
import P2.mlutil as mlutil
import P2.crossval as crossval


def main(path: str, skip_cv: bool, algorithm, iterations, save_as, experiment):
    file_base, root_dir = mainutil.get_dataset_and_path(path)
    data = mldata.parse_c45(file_base=file_base, rootdir=root_dir)
    n_folds = 1 if skip_cv else 5
    learner = Adaboost(algorithm=algorithm, experiment=experiment)
    predictions, labels = crossval.cross_validate(learner=learner, data=data, n_folds=n_folds, save_as=save_as)
    mainutil.get_results(is_experiment=0, predictions=predictions, labels=labels, print_results=1, n_folds=n_folds)


class Adaboost(model.Model):
    def __init__(self, algorithm, experiment=0, iterations=10, models=None, cweights=None, accuracies=None):
        if experiment:
            self.algorithm = ['dtree', 'nbayes', 'logreg']
            if experiment == 2:
                iterations = iterations / 3
        else:
            self.algorithm = [algorithm]
        self.iterations = iterations
        self.model = models
        self.cweights = cweights
        self.accuracies = accuracies
        self.experiment = experiment

    def get_name(self):
        return self.__class__.__name__

    def save(self, file: str) -> NoReturn:
        with open(file, 'w') as f:
            saved = {
                'models': self.model,
                'cweights': self.cweights,
                'algorithm': self.algorithm
            }

    #            json.dump(saved, f, indent='\t')

    @staticmethod
    def load(file: str):
        with open(file) as f:
            learner = json.load(f)
            models = learner['learners']
            cweights = learner['cweights']
            algorithm = learner['algorithm']
        return Adaboost(algorithm=algorithm, models=models, cweights=cweights)

    def train(self, data):  # initialize example weights to 1/N. where each data entry is a feature
        weights = [1 / len(data)] * len(data)  # initialize example weights to 1/N. where each data entry is a feature
        next_weights = [1 / len(data)] * len(data)

        i = 0
        epsilon = 0.5
        accuracies = []
        model = []  # the models themselves
        cweights = []  # the classifier weights
        # run model and get predictions
        # note that we're training and testing on the same data
        labels = mlutil.get_labels(data)
        while i < self.iterations and 0 < epsilon <= 0.5:
            classifier = 0
            classifier_weight = 0
            epsilon = 0
            acc = 0
            # run through all algorithms we want per iteration
            for e, alg in enumerate(self.algorithm):
                weights = copy(next_weights)
                mod = weighted_model(data, weights, alg=alg)
                pred = mod.predict(data)
                # this returns scores for the models, and labels for decision tree
                weights1, classifier_weight1, epsilon1, acc1 = weighted_error(pred, labels, weights)  # find new example weights and classifier weight
                if self.experiment == 2: # experiment where we use all classifiers
                    model.append(mod)
                    cweights.append(classifier_weight1)
                    accuracies.append(acc)
                if acc1 > acc:
                    next_weights = weights1
                    classifier_weight = classifier_weight1
                    epsilon = epsilon1
                    acc = acc1
                    classifier = mod
            if self.experiment < 2:
                model.append(classifier)
                cweights.append(classifier_weight)
                accuracies.append(acc)
            i = i + 1
        self.model = model
        self.cweights = cweights
        self.accuracies = accuracies

    def predict(self, data):
        # get the final output by weighted output
        final_pred = np.zeros(len(data))
        if self.experiment:
            weights = self.accuracies
        else:
            weights = self.cweights
        tot_weights = sum(weights)
        for i, m in enumerate(self.model):
            plabels = m.predict(data)
            final_pred = final_pred + (np.multiply(plabels, weights[i])) / tot_weights
        return final_pred
    # predict using ALL classifiers


def weighted_error(plabels, truths, weights):
    plabels = [p > 0.5 for p in plabels]
    ind = [i for i, e in enumerate(truths) if e != plabels[i]]  # save the indices that are wrong
    acc = len(ind) / len(plabels)
    epsilon = min(1, sum([weights[i] for i in ind]))  # resolve rounding error
    classifier_weight = 0.5 * np.log((1 - epsilon) / epsilon)
    Z = sum(weights)
    weights = [w * np.exp(-w * truths[i] * plabels[i] / Z) for i, w in enumerate(weights)]
    return weights, classifier_weight, epsilon, acc


def weighted_model(data: ExampleSet, weights, alg):
    if alg == 'dtree':
        learner = P1.algorithm.ID3(boost_weights=weights, split_function=metrics.info_gain, max_depth=1)
        learner.train(data)
        # or i could directly call the learner, id3
    if alg == 'nbayes':
        learner = nbayes.NaiveBayes(n_bins=4, laplace_m=2, boost_weights=weights)
    if alg == 'logreg':
        learner = logreg.LogisticRegression(iterations=1000, step_size=0.5, boost_weights=weights)
    learner.train(data)
    return learner

if __name__ == "__main__":
    random.seed(a=12345)
    # command_line_main()
    #algorithm = 'nbayes'
    algorithm = 'logreg'
    #algorithm = 'dtree'
    iterations = 10
    path = '..\\voting'
    main(path=path, skip_cv=False, algorithm=algorithm, iterations = iterations, save_as=path + algorithm, experiment=2)
