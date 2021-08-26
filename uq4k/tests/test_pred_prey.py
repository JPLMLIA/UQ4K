# Tests with pytest for the classes and methods in pred_prey.py
#
# Author   : Mike Stanley
# Written  : August 26, 2021
# Last Mod : August 26, 2021

from pred_prey import PredPrey

# Set up a predator-prey model
ALPHA, BETA, DELTA, GAMMA = 0.55, 0.025, 0.02, 0.8
PREY_INIT, PRED_INIT = 30, 10
eval_times_hr = np.linspace(start=0, stop=20, num=200)
predprey = PredPrey(
    alpha=ALPHA, alpha_bounds=(-5, 5),
    gamma=GAMMA, gamma_bounds=(-5, 5),
    beta=BETA, delta=DELTA,
    prey_init=PREY_INIT, pred_init=PRED_INIT,
    time_idx=eval_times_hr
)

def test_call():
    """ Tests the forward model capabilities of the pred-prey model """
    assert isinstance(
        predprey(np.array([ALPHA, GAMMA]),
        np.arrya
    )
