import time
import os
import logging
from multiprocessing import Process, Queue, Lock, Manager
from trial import Trial, TrialConfig

class Scheduler:
    """
    The Scheduler abstract class defines the basic structure of a scheduler, which must
    schedule and run the trials that constitute hyperparameter optimization.
    """
    def __init__(self, experiment_config, hyperparameter_space, logger) -> None:
        self.experiment_config = experiment_config
        self.hyperparameter_space = hyperparameter_space
        self.logger = logger

    def run(self):
        raise NotImplementedError("Scheduler is an abstract class, please subclass")

class ParallelRoundRobinScheduler(Scheduler):
    """
    The ParallelRoundRobinScheduler is a Scheduler that starts a consumer for each
    machine, and a single producer that yields hyperparameter configurations (each will
    have its own trial). Each consumer takes the next available hyperparameter
    configuration, executes the corresponding trial, then takes the next available
    configuration from the producer. All consumers execute in parallel (using
    multiprocessing), and take configurations in what is essentially a round-robin
    grid search.
    """
    def __init__(self, experiment_config, hyperparameter_space, logger) -> None:
        super(ParallelRoundRobinScheduler, self).__init__(experiment_config, hyperparameter_space, logger)

    def producer(self, queue, lock, results):
        with lock:
            self.logger.debug(f"Starting producer with PID {os.getpid()}")

        for hyperparameter_config in self.hyperparameter_space:
            queue.put(hyperparameter_config)

        # Since there is only 1 producer, wait until all spaces have been processed before returning
        while len(results) != len(self.hyperparameter_space):
            time.sleep(1)

        with lock:
            self.logger.debug=(f"Producer {os.getpid()} exiting...")

    def consumer(self, queue, lock, machine, results):
        with lock:
            self.logger.debug(f"Assigning machine {machine} to PID {os.getpid()}")

        while True:
            # Block until the queue has a hyperparameter config to retrieve
            hyperparameter_config = queue.get()
            trial_config = TrialConfig(machine, self.experiment_config)
            trial = Trial(trial_config, hyperparameter_config, self.logger)

            with lock:
                self.logger.debug(f"{machine} now training {hyperparameter_config.get_dict()}...")

            trial_result = trial.run()

            # Add output to shared results list
            results.append(trial_result)

            with lock:
                self.logger.debug(f"{machine} finished training {hyperparameter_config.get_dict()}")

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

        self.logger.debug("All trials have now been executed!")
        return results

