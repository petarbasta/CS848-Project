import os
import subprocess

class TrialConfig:
    def __init__(self, machine, username, venv_dir, train_file) -> None:
        self.machine = machine
        self.username = username
        self.venv_dir = venv_dir
        self.train_file = train_file

class Trial:
    def __init__(self, trial_config, hyperparameter_config) -> None:
        self.machine = trial_config.machine
        self.hyperparameter_config = hyperparameter_config

        venv_cmd = f"source {os.path.join(trial_config.venv_dir, 'bin/activate')}"
        train_args = ' '.join(f"--{name} \"{value}\"" for name, value in hyperparameter_config.get_dict().items())
        train_cmd = f"python {trial_config.train_file} {train_args}"
        ssh_cmd = f"ssh {trial_config.username}@{trial_config.machine} -o StrictHostKeyChecking=no"
        self.full_cmd = f"{ssh_cmd} \"{venv_cmd} && {train_cmd}\""

    def run(self):
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] Logging into {self.machine} and beginning training...")
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] {self.full_cmd}")
        result = subprocess.run(self.full_cmd, shell=True, universal_newlines=True, stdout=subprocess.PIPE)
        output = '\n'.join(result.stdout.splitlines())
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] Recieved training output: {output}")

        # TODO: Parse and return results (which should be encoded in standardized format, perhaps JSON)
        return output

