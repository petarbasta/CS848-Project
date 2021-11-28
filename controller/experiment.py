import json
from scheduler import ParallelRoundRobinScheduler

class ExperimentConfig:
    def __init__(self, controller_args) -> None:
        self.remote_username = controller_args.username
        self.remote_machines = controller_args.machines
        self.venv_dir = controller_args.venv
        self.train_file = controller_args.dnn
        self.dnn_metric_key = controller_args.dnn_metric_key
        with open(controller_args.dnn_train_args) as f:
            args_dict = json.load(f)
            self.train_args  = args_dict

class Experiment:
    def __init__(self, experiment_config, hyperparameter_space) -> None:
        self.scheduler = ParallelRoundRobinScheduler(experiment_config, hyperparameter_space)

    def run(self):
        trial_results = self.scheduler.run()
        return trial_results

