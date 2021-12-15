from scheduler import ParallelRoundRobinScheduler

class ExperimentConfig:
    def __init__(self, controller_args) -> None:
        self.remote_username = controller_args.username
        self.remote_password = controller_args.password
        self.remote_machines = controller_args.machines
        self.venv_dir = controller_args.venv
        self.train_file = controller_args.dnn
        self.dnn_metric_key = controller_args.dnn_metric_key
        self.train_args = {
            'arch': controller_args.arch,
            'parallelism': controller_args.parallelism,
            'epochs': controller_args.epochs
        }
        if controller_args.data is not None:
            self.train_args['data'] = controller_args.data

class Experiment:
    """
    The Experiment abstract class defines the basic structure of a experiment, which runs
    a specific scheduler to perform hyperparameter optimization.
    """
    def __init__(self, experiment_config, hyperparameter_space, logger) -> None:
        self.experiment_config = experiment_config
        self.hyperparameter_space = hyperparameter_space
        self.logger = logger

    def run(self):
        raise NotImplementedError("Experiment is an abstract class, please subclass")

class GridSearchExperiment(Experiment):
    """
    The GridSearchExperiment is an Experiment which runs hyperparameter optimization
    using a basic grid search algorithm, which is executed using a parallel round-robin
    scheduler.
    """
    def __init__(self, experiment_config, hyperparameter_space, logger) -> None:
        super(GridSearchExperiment, self).__init__(experiment_config, hyperparameter_space, logger)
        self.scheduler = ParallelRoundRobinScheduler(experiment_config, hyperparameter_space, logger)

    def run(self):
        trial_results = self.scheduler.run()
        return trial_results

