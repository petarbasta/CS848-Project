#!/usr/bin/env python

import argparse
import json
import sys
import logging
import ray
from ray import tune
from ray.tune.integration.horovod import DistributedTrainableCreator
from train import run_horovod_raytune_mnist_training

"""
python horovod_raytune.py \
        --arch resnet \
        --epochs 1 \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/models/MNIST/hyperparameter_space_MNIST.json \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max
"""

supported_model_architectures = ['resnet', 'alexnet']


def init_args():
    parser = argparse.ArgumentParser(description='Horovod+RayTune PyTorch Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet', required=True,
                        choices=supported_model_architectures,
                        help='model architecture: ' +
                            ' | '.join(supported_model_architectures) +
                            ' (default: resnet)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--dnn_hyperparameter_space', required=True, type=str, help='The JSON file defining the DNN hyperparameter space')
    parser.add_argument('--dnn_metric_key', required=True, type=str, help='The key for the relevant metric to extract from DNN JSON output')
    parser.add_argument('--dnn_metric_objective', required=True, choices=['max', 'min'], help='Whether to maximize or minimize the metric')
    #parser.add_argument('--debug', help="Print all debugging statements", action="store_const",
    #        dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    #parser.add_argument('--verbose', help="Print all logging statements", action="store_const", dest="loglevel", const=logging.INFO)
    
    args = parser.parse_args()
    return args


def init_hyperparameter_space(path):
    with open(path) as f:
        hyp_dict = json.load(f)
        return {key: tune.grid_search(hyp_dict[key]) for key in hyp_dict}


def main():
    args = init_args()
    hyperparameter_space_dict = init_hyperparameter_space(args.dnn_hyperparameter_space)

    train_executable = tune.with_parameters(run_horovod_raytune_mnist_training,
            arch=args.arch, epochs=args.epochs, dry_run=args.dry_run)

    trainable = DistributedTrainableCreator(
            train_executable,
            #num_cpus_per_slot=1, # 1 CPU for each Ray worker
            num_hosts=1, # Each trial runs on 1 machine
            num_slots=2, # Each trial will employ 2 workers (GPUs)
            use_gpu=True)

    # Run grid search (which is the default optimization strategy)
    ray.init(address='auto')
    analysis = tune.run(
            trainable,
            metric=args.dnn_metric_key, # TODO: Pass this key to trainable as regular param
            mode=args.dnn_metric_objective,
            #name="MyExperiment", # Name the experiment
            #num_samples=1, # Irrelevant when used with grid search
            #verbose=3, # 0,1,2,3, where 0 is least noisy
            config=hyperparameter_space_dict,
            sync_config=tune.SyncConfig(
                syncer=None # Disable syncing when all nodes share filesystem
            ))

    # Print final results
    print(analysis.best_config)


if __name__ == '__main__':
    main()

