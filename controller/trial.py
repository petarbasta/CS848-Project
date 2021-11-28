import os
import paramiko
import json

class TrialConfig:
    def __init__(self, machine, username, password, venv_dir, train_file, train_args, dnn_metric_key) -> None:
        self.machine = machine
        self.username = username
        self.password = password
        self.venv_dir = venv_dir
        self.train_file = train_file
        self.train_args = train_args
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
        reg_args = ' '.join(f"--{name} \"{value}\"" for name, value in trial_config.train_args.items())
        hyp_args = ' '.join(f"--{name} \"{value}\"" for name, value in hyperparameter_config.get_dict().items())
        train_args = f"{reg_args} {hyp_args}"
        train_cmd = f"python {trial_config.train_file} {train_args}"
        self.full_cmd = f"{venv_cmd} && {train_cmd}"

    def run(self):
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] Logging into {self.machine} and beginning training...")
        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] {self.full_cmd}")

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=self.trial_config.machine,
                username=self.trial_config.username, password=self.trial_config.password)
        ssh_client.get_transport().set_keepalive(60)

        stdin, stdout, stderr = ssh_client.exec_command(self.full_cmd)
        output = stdout.readlines()[-1]
        ssh_client.close()

        print(f"[TRIAL.PY {self.machine} {self.hyperparameter_config.get_dict()}] Recieved training output: {output}")

        statistics_dict = json.loads(output)
        reported_metric = float(statistics_dict[self.trial_config.dnn_metric_key])
        reported_runtime = statistics_dict['runtime']
        return TrialResult(
            self.hyperparameter_config,
            reported_metric,
            reported_runtime
        )

