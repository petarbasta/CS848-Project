#!/usr/bin/env python

import argparse
import json
import sys
import logging
#from getpass import getpass
import ray
from ray import tune
from ray.tune.integration.horovod import DistributedTrainableCreator
from models.imagenet.train_horovod_raytune import run_horovod_raytune_imagenet_training
from models.MNIST.train import run_horovod_raytune_mnist_training

"""
Example usage for ImageNet training task:

python horovod_raytune/controller.py \
        --task imagenet \
        --data /u4/jerorseth/datasets/ILSVRC/Data/CLS-LOC \
        --arch resnet \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/imagenet/hyperparameter_space_ImageNet.json \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max


Example usage for MNIST training task:

python horovod_raytune/controller.py \
        --task mnist \
        --arch resnet \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/MNIST/hyperparameter_space_MNIST.json \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max
"""

supported_tasks = ['imagenet', 'mnist']
supported_model_architectures = ['resnet', 'alexnet']


def init_args():
    parser = argparse.ArgumentParser(description='Horovod+RayTune PyTorch Training')
    parser.add_argument('-d', '--data', metavar='DIR', required=False, help='path to dataset (imagenet only)')
    parser.add_argument('-t', '--task', metavar='TASK', default=supported_tasks[0], required=True,
                        choices=supported_tasks,
                        help=f"training task: {' | '.join(supported_tasks)} (default: {supported_tasks[0]})")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet', required=True,
                        choices=supported_model_architectures,
                        help='model architecture: ' +
                            ' | '.join(supported_model_architectures) +
                            ' (default: resnet)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass (mnist only)')
    #parser.add_argument('--username', required=True, type=str, help='Username for SSH to remote machines')
    #parser.add_argument('--password', required=False, type=str, help='Password for SSH to remote machines')
    #parser.add_argument('--machines', required=True, nargs='+', help='All remote machines to utilize')
    #parser.add_argument('--venv', required=True, type=str, help='The venv directory')
    #parser.add_argument('--dnn', required=True, type=str, help='The Python file containing the PyTorch DNN training job')
    parser.add_argument('--dnn_hyperparameter_space', required=True, type=str, help='The JSON file defining the DNN hyperparameter space')
    #parser.add_argument('--dnn_train_args', required=True, type=str, help='The JSON file defining arguments to pass to DNN training script')
    parser.add_argument('--dnn_metric_key', required=True, type=str, help='The key for the relevant metric to extract from DNN JSON output')
    parser.add_argument('--dnn_metric_objective', required=True, choices=['max', 'min'], help='Whether to maximize or minimize the metric')
    #parser.add_argument('--debug', help="Print all debugging statements", action="store_const",
    #        dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    #parser.add_argument('--verbose', help="Print all logging statements", action="store_const", dest="loglevel", const=logging.INFO)
    
    args = parser.parse_args()
    #if not args.password:
    #    args.password = getpass('Password for SSH to remote machines:')
    return args


def init_hyperparameter_space(path):
    with open(path) as f:
        hyp_dict = json.load(f)
        return {key: tune.grid_search(hyp_dict[key]) for key in hyp_dict}


def main():
    args = init_args()
    hyperparameter_space_dict = init_hyperparameter_space(args.dnn_hyperparameter_space)

    """
    # Create logger that will be shared with all modules
    logger = logging.getLogger(__name__)
    logger.setLevel(args.loglevel)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setLevel(args.loglevel)
    log_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.addHandler(log_handler)
    """

    if args.task == 'imagenet':
        train_executable = tune.with_parameters(run_horovod_raytune_imagenet_training,
            data=args.data, arch=args.arch, epochs=args.epochs)
    elif args.task == 'mnist':
        train_executable = tune.with_parameters(run_horovod_raytune_mnist_training,
            arch=args.arch, epochs=args.epochs, dry_run=args.dry_run)

    trainable = DistributedTrainableCreator(
            train_executable,
            num_cpus_per_slot=1, # 1 CPU for each Ray worker
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

