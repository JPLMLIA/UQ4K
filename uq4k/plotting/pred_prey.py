# Plotting functions for the predator/prey example
#
# Author   : Mike Stanley
# Written  : Sept 23, 2021
# Last Mod : Sept 23, 2021

import matplotlib.pyplot as plt
import numpy as np

def plot_dyn_and_data(evals_times, pop_dyn, data):
    """
    Plot the true population dynamics and noisy observations.

    Parameters:
    -----------
        evals_times (np arr) : time points at which the model is run
        pop_dyn     (np arr) : true population dynamics
        data        (np arr) : observed data

    Returns:
    --------
        None -- but makes plot!
    """
    plt.figure(figsize=(9.5, 4.5))

    # plot the noisy observations
    plt.plot(evals_times, data[:, 0], color="red", label=r"Prey Component: $x_t$")
    plt.plot(evals_times, data[:, 1], color="blue", label=r"Predator Component: $y_t$")

    # plot the true Dynamics
    plt.plot(
        evals_times,
        pop_dyn[:, 0],
        color="red",
        label="True Prey Population",
        linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        evals_times,
        pop_dyn[:, 1],
        color="blue",
        label="True Predator Population",
        linestyle="--",
        alpha=0.5,
    )

    # axis labels
    plt.ylabel("Population Counts")
    plt.xlabel("$t$ (time)")
    plt.ylim(0, 80)

    # change x labels to be integers
    plt.xticks(np.arange(0, 21, 2))

    plt.legend()
    plt.tight_layout()
    plt.show()