import os
import paramiko
import json

class TrialConfig:
    def __init__(self, machine, experiment_config) -> None:
        self.machine = machine
        self.username = experiment_config.remote_username
        self.password = experiment_config.remote_password
        self.venv_dir = experiment_config.venv_dir
        self.train_file = experiment_config.train_file
        self.train_args = experiment_config.train_args
        self.dnn_metric_key = experiment_config.dnn_metric_key

class TrialResult:
    def __init__(self, hyperparameter_config, value, runtime, mem_params, mem_bufs, mem_peak) -> None:
        self.hyperparameter_config = hyperparameter_config
        self.value = value
        self.runtime = runtime
        self.mem_params = mem_params
        self.mem_bufs = mem_bufs
        self.mem_peak = mem_peak

    def __str__(self):
        return f"{self.hyperparameter_config.get_dict()}: {self.value}, runtime: {self.runtime}, mem_params: {self.mem_params}, mem_bufs: {self.mem_bufs}, mem_peak: {self.mem_peak}"

class Trial:
    def __init__(self, trial_config, hyperparameter_config, logger) -> None:
        self.machine = trial_config.machine
        self.hyperparameter_config = hyperparameter_config
        self.trial_config = trial_config
        self.logger = logger

        venv_cmd = f"source {os.path.join(trial_config.venv_dir, 'bin/activate')}"
        reg_args = ' '.join(f"--{name} \"{value}\"" for name, value in trial_config.train_args.items())
        hyp_args = ' '.join(f"--{name} \"{value}\"" for name, value in hyperparameter_config.get_dict().items())
        train_args = f"{reg_args} {hyp_args}"
        train_cmd = f"python {trial_config.train_file} {train_args}"
        self.full_cmd = f"{venv_cmd} && {train_cmd}"

    def run(self):
        self.logger.debug(f"Attempting to establish SSH connection with {self.machine}...")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=self.trial_config.machine,
                username=self.trial_config.username, password=self.trial_config.password)
        ssh_client.get_transport().set_keepalive(60)
        self.logger.debug(f"SSH connection with {self.machine} established successfully!")

        self.logger.debug(f"Running cmd on {self.machine}: {self.full_cmd}")
        self.logger.info(f"Training {self.hyperparameter_config.get_dict()} on {self.machine}...")
        stdin, stdout, stderr = ssh_client.exec_command(self.full_cmd)
        output_lines = stdout.readlines()
        self.logger.debug(f"Received output from {self.machine}: {output_lines}")
       
        output_errors = stderr.readlines()
        self.logger.debug(f"Received error from {self.machine}: {output_errors}")
 
        ssh_client.close()
        self.logger.debug(f"SSH connection with {self.machine} has been closed")

        output = output_lines[-1].strip()
        self.logger.info(f"{self.hyperparameter_config.get_dict()} => {output}")

        statistics_dict = json.loads(output)
        reported_metric = float(statistics_dict[self.trial_config.dnn_metric_key])
        reported_runtime = statistics_dict['runtime']
        reported_mem_params = statistics_dict['mem_params']
        reported_mem_bufs = statistics_dict['mem_bufs']
        reported_mem_peak = statistics_dict['mem_peak']

        return TrialResult(
            self.hyperparameter_config,
            reported_metric,
            reported_runtime,
            reported_mem_params,
            reported_mem_bufs,
            reported_mem_peak
        )


