import time
import os
from multiprocessing import Process, Queue, Lock, Manager
from trial import Trial, TrialConfig

class ParallelRoundRobinScheduler:
    def __init__(self, experiment_config, hyperparameter_space) -> None:
        self.experiment_config = experiment_config
        self.hyperparameter_space = hyperparameter_space

    def producer(self, queue, lock, results):
        with lock:
            print(f"[SCHEDULER.PY] Starting producer with PID {os.getpid()}")

        for hyperparameter_config in self.hyperparameter_space:
            queue.put(hyperparameter_config)

        # Since there is only 1 producer, wait until all spaces have been processed before returning
        while len(results) != len(self.hyperparameter_space):
            time.sleep(1)

        with lock:
            print(f"[SCHEDULER.PY] Producer {os.getpid()} exiting...")

    def consumer(self, queue, lock, machine, results):
        with lock:
            print(f"[SCHEDULER.PY] Assigning machine {machine} to PID {os.getpid()}")

        while True:
            # Block until the queue has a hyperparameter config to retrieve
            hyperparameter_config = queue.get()
            
            trial_config = TrialConfig(machine, self.experiment_config.remote_username,
                    self.experiment_config.venv_dir, self.experiment_config.train_file,
                    self.experiment_config.dnn_metric_key)
            trial = Trial(trial_config, hyperparameter_config)

            with lock:
                print(f"[SCHEDULER.PY] {machine} is now training hyperparameter config {hyperparameter_config.get_dict()}...")

            trial_result = trial.run()

            # Add output to shared results list
            results.append(trial_result)

            with lock:
                print(f"[SCHEDULER.PY] {machine} has finished training hyperparameter config {hyperparameter_config.get_dict()}")

    def run(self):
        queue = Queue()
        lock = Lock()
        manager = Manager()

        producers = []
        consumers = []
        results = manager.list()

        producers.append(Process(target=self.producer, args=(queue, lock, results)))

        for machine in self.experiment_config.remote_machines:
            p = Process(target=self.consumer, args=(queue, lock, machine, results))

            # Set daemon to true to allow producer to dictate termination of consumers
            p.daemon = True
            consumers.append(p)

        # Launch independent processes for each consumer and producer process
        for p in producers:
            p.start()
        for c in consumers:
            c.start()

        for p in producers:
            p.join()

        print("[SCHEDULER.PY] All trials have been executed")
        return results

