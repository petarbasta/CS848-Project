import os
import subprocess
import json
from timeit import default_timer as timer

class TrialConfig:
    def __init__(self, machine, username, venv_dir, train_file, dnn_metric_key) -> None:
        self.machine = machine
        self.username = username
        self.venv_dir = venv_dir
        self.train_file = train_file
        self.dnn_metric_key = dnn_metric_key

class TrialResult:
    def __init__(self, hyperparameter_config, value, runtime) -> None:
        self.hyperparameter_config = hyperparameter_config
        self.value = value
        self.runtime = runtime

    def __str__(self):
        return f"{self.hyperparameter_config.get_dict()}: {self.value}, runtime: {self.runtime}"

class Trial:
    def __init__(self, trial_config, hyperparameter_config) -> None:
        self.machine = trial_config.machine
        self.hyperparameter_config = hyperparameter_config
        self.trial_config = trial_config

        venv_cmd = f"source {os.path.join(trial_config.venv_dir, 'bin/activate')}"
        train_args = ' '.join(f"--{name} \"{value}\"" for name, value in hyperparameter_config.get_dict().items())
        train_cmd = f"python {trial_config.train_file} {train_args}"
        ssh_cmd = f"ssh {trial_config.username}@{trial_config.machine} -o StrictHostKeyChecking=no"
        self.full_cmd = f"{ssh_cmd} \"{venv_cmd} && {train_cmd}\""

    def run(self):
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] Logging into {self.machine} and beginning training...")
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] {self.full_cmd}")

        start_time = timer()
        result = subprocess.run(self.full_cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE)
        end_time = timer()
        runtime_s = end_time - start_time

        output = '\n'.join(result.stdout.splitlines())
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] Recieved training output: {output}")

        return TrialResult(
            self.hyperparameter_config,
            float(json.loads(output)[self.trial_config.dnn_metric_key]),
            runtime_s
        )

