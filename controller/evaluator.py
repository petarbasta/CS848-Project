from operator import attrgetter

class Evaluator:
    def __init__(self, minimize):
        self.should_minimize = minimize

    def get_best(self, trial_results):
        if self.should_minimize:
            return min(trial_results, key=attrgetter('value'))
        else:
            return max(trial_results, key=attrgetter('value'))

