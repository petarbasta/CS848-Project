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
    plt.ylabel("Peak Memory (Gigabytes")
    plt.title(
        f"ImageNet on AlexNet peak memory by trial and parallelization strategy")

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
            {"runtime":  12357.977592938987 , "mem_peak": 848619520/1073741824 },
            {"runtime": 12358.222774202004 , "mem_peak": 1255090176/1073741824 },
            {"runtime": 12359.53697986342 , "mem_peak": 2071078912/1073741824 },
            {"runtime": 12199.161570680008 , "mem_peak": 848619520/1073741824 },
            {"runtime": 12199.465173619334 , "mem_peak": 1255090176/1073741824 },
            {"runtime": 12200.772594962269 , "mem_peak": 2071078912/1073741824 },
        ],
        'DP': [
            {"runtime": 10911.633804809302 , "mem_peak": 1562935808/1073741824 },
            {"runtime": 10910.587639587 , "mem_peak": 1469010944/1073741824 },
            {"runtime": 10909.30910898 , "mem_peak": 1969908736/1073741824 },
            {"runtime": 11028.597851529717 , "mem_peak": 1469010944/1073741824 },
            {"runtime": 11027.396494140994 , "mem_peak": 1969908736/1073741824 },
            {"runtime": 11834.253983154893 , "mem_peak": 1562935808/1073741824 },
        ],
        # 'GPipe': [
        #     {"runtime":  , "mem_peak":  },
        #     {"runtime":  , "mem_peak":  },
        #     {"runtime":  , "mem_peak":  },
        #     {"runtime":  , "mem_peak":  },
        #     {"runtime":  , "mem_peak":  },
        #     {"runtime":  , "mem_peak":  },
        # ],
        'Horovod + Ray Tune': [
            {"runtime": 6080.12 , "mem_peak": 1705150976/1073741824 },
            {"runtime": 6018.94 , "mem_peak": 2111747072/1073741824 },
            {"runtime": 6079.34 , "mem_peak": 2793867264/1073741824 },
            {"runtime": 6149.93 , "mem_peak": 1704577536/1073741824 },
            {"runtime": 6155.93 , "mem_peak": 1977529344/1073741824 },
            {"runtime": 6156.2 , "mem_peak": 2793867264/1073741824 },
        ],

    }
    plot_trial_results(all_trial_results, num_trials, 'mem_peak', 's')

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
