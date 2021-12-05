#!/usr/bin/env python

import argparse
import json
import sys
import logging
#from getpass import getpass
from ray import tune
from ray.tune.integration.horovod import DistributedTrainableCreator
from train import run_training

"""
python horovod_raytune/controller.py \
        --data /u4/jerorseth/datasets/ILSVRC/Data/CLS-LOC \
        --arch resnet \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/horovod_raytune/hyperparameter_space_resnet.json \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max
"""

supported_model_architectures = ['resnet']


def init_args():
    parser = argparse.ArgumentParser(description='Horovod+RayTune PyTorch ImageNet Training')
    parser.add_argument('-d', '--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=supported_model_architectures,
                        help='model architecture: ' +
                            ' | '.join(supported_model_architectures) +
                            ' (default: resnet)')
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

    trainable = DistributedTrainableCreator(
            tune.with_parameters(run_training, data=args.data, arch=args.arch),
            num_cpus_per_slot=1, # 1 CPU for each Ray worker
            num_hosts=1, # Each trial runs on 1 machine
            num_slots=2, # Each trial will employ 2 workers (GPUs)
            use_gpu=True)
    # TODO: More arguments can be specified, see https://docs.ray.io/en/latest/tune/api_docs/execution.html
    # Run grid search (which is the default optimization strategy)
    print(hyperparameter_space_dict)
    analysis = tune.run(
            trainable,
            metric=args.dnn_metric_key, # TODO: Pass this key to trainable as regular param
            mode=args.dnn_metric_objective,
            #name="MyExperiment", # Name the experiment
            #num_samples=1, # Irrelevant when used with grid search
            #verbose=3, # 0,1,2,3, where 0 is least noisy
            config=hyperparameter_space_dict)

    print(analysis.best_config)


if __name__ == '__main__':
    main()

