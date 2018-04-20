#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import shutil
import pickle
import time
import argparse

import torch
import numpy as np
from IPython.core import ultratb
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.goodbooks import get_goodbooks_dataset
from spotlight.datasets.amazon import get_amazon_dataset
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import sequence_mrr_score
from spotlight.torch_utils import set_seed
import hyperopt
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

from mlstm4reco.representations import mLSTMNet


CUDA = torch.cuda.is_available()

# Start IPython shell on exception (nice for debugging)
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

# First element is always default
DATASETS = ['1m', '10m', 'amazon', 'goodbooks']
MODELS = ['mlstm', 'lstm']


def parse_args(args):
    parser = argparse.ArgumentParser(description='Run experiment for benchmarks.')
    parser.add_argument('dataset',
                        type=str,
                        choices=DATASETS,
                        default=DATASETS[0],
                        help='Name of the dataset or variant for Movielens')
    parser.add_argument('-n', '--num_trials',
                        dest='num_trials',
                        default=100,
                        type=int,
                        help='Number of trials to run')
    parser.add_argument('-m', '--model',
                        dest='model',
                        default=MODELS[0],
                        choices=MODELS,
                        type=str,
                        help='Model for experiment')

    return parser.parse_args(args)


def hyperparameter_space(model_type):
    """Define hyperopt hyperparameter space

    Args:
        model_type: Restrict to model if provided.

    Returns:
        hyperparameter object
    """

    common_space = {
        'batch_size': hp.quniform('batch_size', 64, 256, 16),
        'learn_rate': hp.loguniform('learn_rate', -6, -3),
        'l2': hp.loguniform('l2', -25, -9),
        'n_iter': hp.quniform('n_iter', 20, 50, 5),
        'loss': hp.choice('loss', ['adaptive_hinge']),
        'embedding_dim': hp.quniform('embedding_dim', 16, 128, 8),
    }

    models = [
            {
                'type': 'lstm',
                'representation': 'lstm',
                **common_space
            },
            {
                'type': 'mlstm',
                **common_space
            }
        ]

    space = [model for model in models if model['type'] == model_type]
    assert len(space) == 1

    return space[0]


def get_objective(train, valid, test, random_state=None):

    def objective(space):
        batch_size = int(space['batch_size'])
        learn_rate = space['learn_rate']
        loss = space['loss']
        n_iter = int(space['n_iter'])
        embedding_dim = int(space['embedding_dim'])
        l2 = space['l2']

        if space['type'] == 'mlstm':
            representation = mLSTMNet(
                train.num_items,
                embedding_dim=embedding_dim)
            model = ImplicitSequenceModel(
                loss=loss,
                batch_size=batch_size,
                representation=representation,
                learning_rate=learn_rate,
                n_iter=n_iter,
                l2=l2,
                use_cuda=CUDA,
                random_state=random_state)
        elif space['type'] == 'lstm':
            representation = space['representation']
            model = ImplicitSequenceModel(
                loss=loss,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                representation=representation,
                learning_rate=learn_rate,
                n_iter=n_iter,
                l2=l2,
                use_cuda=CUDA,
                random_state=random_state)
        else:
            raise ValueError('Unknown model type {}'.format(space.get('type', 'NA')))

        start = time.clock()
        try:
            model.fit(train, verbose=True)
        except ValueError:
            elapsed = time.clock() - start
            return {'loss': 0.0,
                    'status': STATUS_FAIL,
                    'validation_mrr': 0.0,
                    'test_mrr': 0.0,
                    'elapsed': elapsed,
                    'hyper': space}
        elapsed = time.clock() - start
        print(model)

        train_mrr = sequence_mrr_score(
            model,
            train,
            exclude_preceding=True
        ).mean()
        validation_mrr = sequence_mrr_score(
            model,
            valid,
            exclude_preceding=True
        ).mean()
        test_mrr = sequence_mrr_score(
            model,
            test,
            exclude_preceding=True
        ).mean()

        print('MRR {} {}'.format(validation_mrr, test_mrr))

        if np.isnan(validation_mrr):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        return {'loss': -validation_mrr,
                'status': status,
                'train_mrr': train_mrr,
                'validation_mrr': validation_mrr,
                'test_mrr': test_mrr,
                'elapsed': elapsed,
                'hyper': space}
    return objective


def optimize(objective, space, trials_fname=None, max_evals=5):

    if trials_fname is not None and os.path.exists(trials_fname):
        with open(trials_fname, 'rb') as trials_file:
            trials = pickle.load(trials_file)
    else:
        trials = Trials()

    fmin(objective,
         space=space,
         algo=hyperopt.tpe.suggest,
         trials=trials,
         max_evals=max_evals)

    if trials_fname is not None:
        temporary = '{}.temp'.format(trials_fname)
        with open(temporary, 'wb') as trials_file:
            pickle.dump(trials, trials_file)
        shutil.move(temporary, trials_fname)

    return trials


def summarize_trials(trials):
    results = trials.trials
    model_type = results[0]['result']['hyper']['type']

    results = sorted(results, key=lambda x: -x['result']['validation_mrr'])

    if results:
        print('Best {}: {}'.format(model_type, results[0]['result']))

    results = sorted(results, key=lambda x: -x['result']['test_mrr'])

    if results:
        print('Best test {}: {}'.format(model_type, results[0]['result']))


def main(args):
    status = 'available' if CUDA else 'not available'
    print("CUDA is {}!".format(status))
    args = parse_args(args)

    # Fix random_state
    seed = 66
    set_seed(seed)
    random_state = np.random.RandomState(seed)

    max_sequence_length = 100
    min_sequence_length = 20
    step_size = max_sequence_length

    if args.dataset == 'amazon':
        max_sequence_length = 50
        min_sequence_length = 5
        step_size = max_sequence_length
        dataset = get_amazon_dataset()
    elif args.dataset == 'goodbooks':
        dataset = get_goodbooks_dataset()
    else:
        dataset = get_movielens_dataset(args.dataset.upper())

    args.variant = args.dataset
    train, rest = user_based_train_test_split(
        dataset,
        test_percentage=0.2,
        random_state=random_state)
    test, valid = user_based_train_test_split(
        rest,
        test_percentage=0.5,
        random_state=random_state)
    train = train.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)
    test = test.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)
    valid = valid.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)

    print('model: {}, data: {}'.format(args.model, train))

    fname = 'experiment_{}_{}.pickle'.format(args.model, args.dataset)
    objective = get_objective(train, valid, test, random_state)
    space = hyperparameter_space(args.model)

    for iteration in range(args.num_trials):
        print('Iteration {}'.format(iteration))
        trials = optimize(objective,
                          space,
                          trials_fname=fname,
                          max_evals=iteration + 1)

        summarize_trials(trials)


if __name__ == '__main__':
    main(sys.argv[1:])
