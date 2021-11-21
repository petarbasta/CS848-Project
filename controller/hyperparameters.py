import itertools

class HyperparameterSpace:
    """
    The HyperparameterSpace class is simply a wrapper for a hyperparameter dict, which maps the
    name of each defined hyperparameter to a (discrete) list of values to trial. This class
    conforms to the Python iterator protocol, making it an iterable to easily iterate through all
    possible hyperparameter value combinations (each combination is a HyperparameterConfig).
    """

    def __init__(self, hyperparameter_space_dict) -> None:
        self._space_dict = hyperparameter_space_dict
        self.list_permutations = list(itertools.product(*self._space_dict.values()))
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < len(self.list_permutations):
            next_config = HyperparameterConfig(dict(zip(self._space_dict.keys(),self.list_permutations[self.current])))
            self.current += 1
            return next_config
        raise StopIteration

    def __len__(self):
        return len(self.list_permutations)

    def get_dict(self):
        return self._space_dict

    def get_values(self, hyperparameter_name):
        return self._space_dict[hyperparameter_name]


class HyperparameterConfig:
    """A HyperparameterConfig is a specific combination of chosen hyperparameters"""
    def __init__(self, hyperparameter_config_dict) -> None:
        self._config_dict = hyperparameter_config_dict

    def get_dict(self):
        return self._config_dict

    def get_value(self, hyperparameter_name):
        return self._config_dict[hyperparameter_name]

