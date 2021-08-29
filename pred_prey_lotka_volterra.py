# Script to reproduce the predator prey/lotka volterra from section 4
#
# Author   : Mike Stanley
# Written  : August 29, 2021
# Last Mod : August 29, 2021

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from uq4k.models.pred_prey import PredPrey
from uq4k.objective_function import MeritFunc
from uq4k.blackbox.bb_optimizer import BbOpt

plt.style.use('seaborn-white')

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
    plt.figure(figsize=(9.5,4.5))

    # plot the noisy observations
    plt.plot(
        evals_times, data[:, 0],
        color='red', label=r'Prey Component: $x_t$'
    )
    plt.plot(
        evals_times, data[:, 1],
        color='blue', label=r'Predator Component: $y_t$'
    )

    # plot the true Dynamics
    plt.plot(
        evals_times, pop_dyn[:, 0], color='red', 
        label='True Prey Population', linestyle='--', alpha=0.5
    )
    plt.plot(
        evals_times, pop_dyn[:, 1], color='blue', 
        label='True Predator Population', linestyle='--', alpha=0.5
    )

    # axis labels
    plt.ylabel('Population Counts')
    plt.xlabel('$t$ (time)')
    plt.ylim(0, 80)

    # change x labels to be integers
    plt.xticks(np.arange(0, 21, 2))

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # script operational parameters
    PLOT_DATA = True

    # ---------- generate data ---------- 
    eval_times = np.linspace(start=0, stop=20, num=200)

    # initial conditions
    PREY_INIT = 30
    PRED_INIT = 10

    # rate of change params
    ALPHA = 0.55
    BETA = 0.025
    DELTA = 0.02
    GAMMA = 0.8

    # conditions for the parameters of interest
    alpha_init = 5
    gamma_init = 5
    alpha_bounds = (-5, 5)
    gamma_bounds = (-5, 5)

    # estimation noise parameter
    SIGMA = 5

    # initialize the predator/prey model
    pred_prey_mod = PredPrey(
        alpha=alpha_init,
        alpha_bounds=alpha_bounds,
        gamma=gamma_init,
        gamma_bounds=gamma_bounds,
        beta=BETA,
        delta=DELTA,
        prey_init=PREY_INIT,
        pred_init=PRED_INIT,
        time_idx=eval_times
    )

    # generate some data
    THETA_TRUE = np.array([ALPHA, GAMMA])
    pop_dyn_true = pred_prey_mod(THETA_TRUE)
    data = pop_dyn_true + stats.norm(
        loc=0, scale=SIGMA
    ).rvs(size=(eval_times.shape[0], 2))

    if PLOT_DATA:
        plot_dyn_and_data(
            evals_times=eval_times,
            pop_dyn=pop_dyn_true,
            data=data
        )

    # ---------- Optimization ----------
    # set optimization parameters
    EPSILON = 0.0001                   # stopping criterion for odad_min_e_ball algo
    CONF_LEV = 0.95                    # 1 - beta_alpha - i.e., prob not violating
    THETA_INIT = np.array([0.5, 0.5])  # starting point for MLE optimization
    MU = 1e13                          # strength of penalty
    BOUNDS = [[-5, 5]]*2               # variable bounds for diff evol algo
    MAX_IT = 10                        # controls number of steps in ball algo
    
    # create objective function and optimizer objects
    objective_obj = MeritFunc(
        forward_model=pred_prey_mod,
        mu=MU,
        data=data
    )
    optimizer = BbOpt(objective_obj=objective_obj)

    # perform the optimization
    mle_theta, M_alpha, S, center, radius_0 = optimizer.optimize_min_e_ball(
        sigma_2=np.square(SIGMA),
        data=data,
        theta_init=THETA_INIT,
        epsilon_0=EPSILON,
        conf_lev=CONF_LEV,
        man_delta=None,
        bounds=BOUNDS,
        max_iter=MAX_IT
    )

    print('----- Center and Radius -----')
    print(S)
    print(center)
    print(radius_0)

    # perform optimization to find dirac weights
    p_opt = optimizer.weight_optimization(S=S)
    print('----- Dirac Weights -----')
    print(p_opt)