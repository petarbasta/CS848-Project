from scheduler import ParallelRoundRobinScheduler

class ExperimentConfig:
    def __init__(self, controller_args) -> None:
        self.remote_username = controller_args.remote_username
        self.remote_machines = controller_args.remote_machines
        self.venv_dir = controller_args.venv_dir
        self.train_file = controller_args.train_file


class Experiment:
    def __init__(self, experiment_config, hyperparameter_space) -> None:
        self.scheduler = ParallelRoundRobinScheduler(experiment_config, hyperparameter_space)

    def run(self):
        results = self.scheduler.run()
        #print(results)
        return results

