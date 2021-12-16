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
    plt.xlabel("Trial")
    plt.ylabel("Runtime (s)")
    plt.title(
        f"ImageNet on ResNet runtime by trial and parallelization strategy")

    plt.xticks(ind + width / 2, (f"T{i}" for i in range(num_trials)))
    plt.legend(loc='lower right')
    plt.show()


def plot_overall_runtime(overall_runtimes, units='s'):
    x = [k for k, v in overall_runtimes.items()]
    x_pos = [i for i, _ in enumerate(x)]
    y = [v for _, v in overall_runtimes.items()]

    # Plot overall runtime for each parallelization strategy
    plt.bar(x_pos, y)

    y_label = "Runtime" if units is None else f"Runtime ({units})"
    plt.ylabel(y_label)
    plt.title("Overall HPO runtime by parallelization strategy")

    plt.xticks(x_pos, x)
    plt.legend(loc='best')
    plt.show()


def main():
    num_trials = 6
    all_trial_results = {
        'MP': [
            {"runtime": 12538.097169693 , "mem_peak": 5038000128/1073741824 },
            {"runtime": 12538.7607512041 , "mem_peak": 9496807424/1073741824 },
            {"runtime":  12540.225725911558, "mem_peak": 18542872576/1073741824 },
            {"runtime": 12321.795147397002 , "mem_peak": 5038000128/1073741824 },
            {"runtime": 12322.499817493372 , "mem_peak": 9496807424/1073741824 },
            {"runtime": 12324.146937515587 , "mem_peak": 18542872576/1073741824 },
        ],
        'DP': [
            {"runtime": 8943.131643308792 , "mem_peak": 8441763328/1073741824 },
            {"runtime": 8943.656773745002 , "mem_peak": 5537866240/1073741824 },
            {"runtime": 8945.405507232994 , "mem_peak": 14247985152/1073741824 },
            {"runtime": 9166.629523647018 , "mem_peak": 5537866240/1073741824 },
            {"runtime": 9150.694722752 , "mem_peak": 8441763328/1073741824 },
            {"runtime": 9152.837406508625 , "mem_peak": 14247985152/1073741824 },
        ],
        'GPipe': [
            {"runtime": 12160.291532916017 , "mem_peak": 2749442560/1073741824 },
            {"runtime": 12161.974192559719 , "mem_peak": 2749442560/1073741824 },
            {"runtime": 12161.142237305 , "mem_peak": 3182885376/1073741824 },
            {"runtime": 12860.294774234295 , "mem_peak": 2749442560/1073741824 },
            {"runtime": 12859.448222236999 , "mem_peak": 3182898176/1073741824 },
            {"runtime": 12864.480598716997 , "mem_peak": 2749442560/1073741824 },
        ],
        'Horovod + Ray Tune': [
            {"runtime": 5810.28 , "mem_peak": 8675167232/1073741824 },
            {"runtime": 6182.01 , "mem_peak": 14480471552/1073741824 },
            {"runtime": 5814.71 , "mem_peak": 22658058752/1073741824 },
            {"runtime": 5992.32 , "mem_peak": 8675167232/1073741824 },
            {"runtime": 5976.9 , "mem_peak": 14480471552/1073741824 },
            {"runtime": 6482.95 , "mem_peak": 22655961600/1073741824 },
        ],

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
