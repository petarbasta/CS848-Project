import argparse
from experiment import Experiment, ExperimentConfig

"""
python controller.py \
        --venv_dir /u4/jerorset/cs848/CS848-Project/venv \
        --train_file /u4/jerorset/cs848/CS848-Project/controller/fake_dnn.py  \
        --remote_username jerorset \
        --remote_machines gpu1 gpu2 gpu3
"""

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote_username', required=True, type=str, help='Username for SSH to remote machines')
    parser.add_argument('--remote_machines', required=True, nargs='+', help='All remote machines to utilize')
    parser.add_argument('--venv_dir', required=True, type=str, help='The venv directory')
    parser.add_argument('--train_file', required=True, type=str, help='The Python file containing the PyTorch training job')

    args = parser.parse_args()
    return args

def main():
    args = init_args()

    # TODO: Allow hyperparameter space definition from config file
    hyperparameter_space = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10']

    experiment_config = ExperimentConfig(args)
    experiment = Experiment(experiment_config, hyperparameter_space)
    results = experiment.run()

    # TODO: Do something with the results
    print(f"[CONTROLLER.PY] Results: {results}")


if __name__ == '__main__':
    main()

