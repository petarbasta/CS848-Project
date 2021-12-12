import numpy as np
import matplotlib.pyplot as plt


def plot_trial_results(all_trial_results, num_trials, metric_key, units=None):
    ind = np.arange(num_trials)
    width = 0.2
    i = 0

    # Plot trial results for each parallelization strategy
    for parallelization_strat, trial_results in all_trial_results.items():
        plt.bar(ind + (i * width), list(res[metric_key] for res in trial_results),
                width, label=parallelization_strat)
        i += 1

    y_label = metric_key if units is None else f"{metric_key} ({units})"
    plt.ylabel(y_label.capitalize())
    plt.title(f"{metric_key} by trial and parallelization strategy".capitalize())

    plt.xticks(ind + width / 2, (f"T{i}" for i in range(num_trials)))
    plt.legend(loc='best')
    plt.show()


def plot_overall_runtime(overall_runtimes, units='s'):
    x = [k for k,v in overall_runtimes.items()]
    x_pos = [i for i,_ in enumerate(x)]
    y = [v for _,v in overall_runtimes.items()]

    # Plot overall runtime for each parallelization strategy
    plt.bar(x_pos, y)

    y_label = "Runtime" if units is None else f"Runtime ({units})"
    plt.ylabel(y_label)
    plt.title("Overall HPO runtime by parallelization strategy")

    plt.xticks(x_pos, x)
    plt.legend(loc='best')
    plt.show()


def main():
    num_trials = 4
    all_trial_results = {
        'MP': [
            {'accuracy': 13.776000022888184, 'runtime': 12508.823102585971},
            {'accuracy': 17.737998962402344, 'runtime': 12508.739686100977},
            {'accuracy': 9.651999473571777, 'runtime': 12510.027567915618},
            {'accuracy': 3.6419999599456787, 'runtime': 11281.223557015997}
        ],
        'DP': [
            {'accuracy': 15.653999328613281, 'runtime': 12592.37759810104},
            {'accuracy': 13.073999404907227, 'runtime': 12592.40811315598},
            {'accuracy': 12.248000144958496, 'runtime': 12609.93412209116},
            {'accuracy': 4.1579999923706055, 'runtime': 12703.653410563013}
        ],
        'GPipe': [
            {'accuracy': 14.141999244689941, 'runtime': 13139.332205377053},
            {'accuracy': 18.170000076293945, 'runtime': 13140.494633654132},
            {'accuracy': 11.679999351501465, 'runtime': 13154.952540235128},
            {'accuracy': 4.447999954223633, 'runtime': 12652.169604930095}
        ]
    }
    plot_trial_results(all_trial_results, num_trials, 'runtime', 's')

    # Not particularly meaningful
    #plot_trial_results(all_trial_results, num_trials, 'accuracy', '%')

    overall_runtimes = {
        'MP': 23827 / 3600,
        'DP': 25318 / 3600,
        'GPipe': 25840 / 3600,
    }
    #plot_overall_runtime(overall_runtimes, units='hours')


if __name__ == "__main__":
    main()


