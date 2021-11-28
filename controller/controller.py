#!/usr/bin/env python

import argparse
import json
import sys
from experiment import Experiment, ExperimentConfig
from hyperparameters import HyperparameterSpace
from evaluator import Evaluator

"""
python controller.py \
        --venv /u4/jerorset/cs848/CS848-Project/venv \
        --dnn /u4/jerorset/cs848/CS848-Project/models/ImageNet/train.py  \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/models/ImageNet/hyperparameter_space_resnet.json \
        --dnn_train_args /u4/jerorset/cs848/CS848-Project/models/ImageNet/train_args_mp_resnet.json \
        --dnn_metric_key accuracy \
        --dnn_metric_objective max \
        --username jerorset \
        --machines gpu1 gpu2 gpu3
"""

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', required=True, type=str, help='Username for SSH to remote machines')
    parser.add_argument('--machines', required=True, nargs='+', help='All remote machines to utilize')
    parser.add_argument('--venv', required=True, type=str, help='The venv directory')
    parser.add_argument('--dnn', required=True, type=str, help='The Python file containing the PyTorch DNN training job')
    parser.add_argument('--dnn_hyperparameter_space', required=True, type=str, help='The JSON file defining the DNN hyperparameter space')
    parser.add_argument('--dnn_train_args', required=True, type=str, help='The JSON file defining arguments to pass to DNN training script')
    parser.add_argument('--dnn_metric_key', required=True, type=str, help='The key for the relevant metric to extract from DNN JSON output')
    parser.add_argument('--dnn_metric_objective', required=True, choices=['max', 'min'], help='Whether to maximize or minimize the metric')

    args = parser.parse_args()
    return args

def init_hyperparameter_space(path):
    with open(path) as f:
        hyp_dict = json.load(f)
        return HyperparameterSpace(hyp_dict)

def main():
    args = init_args()
    hyperparameter_space = init_hyperparameter_space(args.dnn_hyperparameter_space)

    experiment_config = ExperimentConfig(args)
    experiment = Experiment(experiment_config, hyperparameter_space)
    
    all_trial_results = experiment.run()
    all_printout = "\n".join(str(r) for r in all_trial_results)
    print(f"[CONTROLLER.PY] All Trial Results:")
    print(all_printout)

    evaluator = Evaluator(args.dnn_metric_objective == 'min')
    best_trial_result = evaluator.get_best(all_trial_results) 
    print(f"[CONTROLLER.PY] Best Trial Result:")
    print(str(best_trial_result))


if __name__ == '__main__':
    main()

