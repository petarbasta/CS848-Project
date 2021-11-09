# Considerations:
# 1. To apply MP to PyTorch models, the model layers must be modified to be assigned to devices),
#    or the model must be otherwise reimplemented. For DP, we can use the PyTorch model as-is.
# 2. A specific PyTorchModel subclass will be tightly coupled with a specific associated
#    HyperParameterSpace

class HyperparameterSpace:
    """
    The HyperparameterSpace class is simply a wrapper for a hyperparameter dict, which maps the
    name of each defined hyperparameter to a (discrete) list of values to trial. This class
    conforms to the Python iterator protocol, making it an iterable to easily iterate through all
    possible hyperparameter value combinations (each combination is a HyperparameterConfig).
    """

    def __init__(self, hyperparameter_space_dict) -> None:
        self._space_dict = hyperparameter_space_dict

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: Iterate through self._space_dict and emit each possible HyperparameterConfig
        # Each will look like: { "param1": 372, "param2": 112, "param3": "some_value" }
        # ie.
        # next_val = HyperparameterConfig({ "param1": self._space_dict["param1"][i], ... })
        # return next_val
        # See https://stackoverflow.com/questions/19151/build-a-basic-python-iterator
        pass

    def get_dict(self):
        return self._space_dict

    def get_values(self, hyperparameter_name):
        return self._space_dict[hyperparameter_name]


class HyperparameterConfig:
    """A HyperparameterConfig is a specific combination of chosen hyperparameters"""
    def __init__(self, hyperparameter_config_dict) -> None:
        self._config_dict = hyperparameter_config_dict

    def get_value(self, hyperparameter_name):
        return self._config_dict[hyperparameter_name]


class PyTorchModel:
    """Abstract class representing a PyTorch model"""

    def __init__(self, hyperparameter_config) -> None:
        self.hyperparameter_config = hyperparameter_config
        pass

    def run(self):
        raise NotImplementedError("Instantiate and run a PyTorch model via a subclass")


class MyExampleModel(PyTorchModel):
    """An example model which expects some specific set of hyperparameters"""

    def __init__(self, hyperparameter_config) -> None:
        super(PyTorchModel, self).__init__(hyperparameter_config)

    def run(self):
        # TODO: Implement some actual DNN as a PyTorch model using the config
        # For example:
        # my_model = torch.MyExampleModel(
        #     param1=hyperparameter_config.get_value(param1),
        #     param2=hyperparameter_config.get_value(param2),
        #     ...
        # )
        pass


class Job:
    """
    A Job (aka the top-level experiment) is an abstract class whose subclasses should run a list
    of Trials, using a specific type of hyperparameter search algorithm and employing specific
    types of parallelism (perhaps training-level or trial-level).
    """

    def __init__(self, model, hyperparameter_space) -> None:
        self.model = model
        self.hyperparameter_space = hyperparameter_space

    def run(self):
        raise NotImplementedError("Instantiate and run a Job via a subclass")


class ModelParallelGridSearchJob(Job):
    """
    A ModelParallelGridSearchJob is a Job which uses the Grid Search hyperparameter search
    algorithm, and executes using model parallelism (along with trial parallelism).
    """

    def __init__(self, model, hyperparameter_space, num_workers) -> None:
        # TODO: Determine how many parallel trials vs. parallel workers within trial
        # Here's a basic hardcoded example, where we decide MP parallelism first:
        self.model_parallel_deg = min(num_workers, 3)
        self.parallel_trial_deg = num_workers // self.model_parallel_deg
        self.trials = [
            ModelParallelTrial(model, cfg, self.model_parallel_deg) for cfg in hyperparameter_space
        ]
        super(Job, self).__init__(model, hyperparameter_space)

    def run(self):
        # TODO: Execute parallel_trial_deg trials at a time in parallel until all are exhausted
        # TODO: Implement this within a grid search (we could potentially abstract GridSearch here)
        # ie. my_results = [trial.run() for trial in self.trials]
        pass


class Trial:
    """An abstract Trial whose subclasses must specify a training strategy."""

    def __init__(self, model, hyperparam_config) -> None:
        self.model = model
        self.hyperparam_config = hyperparam_config

    def run(self):
        raise NotImplementedError("Instantiate and run a Trial via a subclass")


class DataParallelTrial(Trial):
    """A DataParallelTrial is a Trial that trains a model using data parallelism."""

    def __init__(self, model, hyperparam_config, num_workers) -> None:
        self.num_workers = num_workers
        super(Trial, self).__init__(model, hyperparam_config)

    def run(self):
        # TODO: Execute data parallel training, do something with results
        pass


class ModelParallelTrial(Trial):
    """A ModelParallelTrial is a Trial that trains a model using model parallelism."""
    def __init__(self, model, hyperparam_config, num_workers) -> None:
        self.num_workers = num_workers
        super(Trial, self).__init__(model, hyperparam_config)

    def run(self):
        # TODO: Execute model parallel training, do something with results
        # NOTE: We must somehow modify the assigned device for each layer to support MP
        pass
