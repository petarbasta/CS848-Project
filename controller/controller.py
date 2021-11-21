import argparse
import json
import sys
from experiment import Experiment, ExperimentConfig
from hyperparameters import HyperparameterSpace

"""
python controller.py \
        --venv /u4/jerorset/cs848/CS848-Project/venv \
        --dnn /u4/jerorset/cs848/CS848-Project/controller/fake_dnn.py  \
        --dnn_hyperparameter_space /u4/jerorset/cs848/CS848-Project/controller/fake_dnn_hyperparameter_space.json \
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
    results = experiment.run()

    # TODO: Do something with the results
    print(f"[CONTROLLER.PY] Results: {results}")


if __name__ == '__main__':
    main()

