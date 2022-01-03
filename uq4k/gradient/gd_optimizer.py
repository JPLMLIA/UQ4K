# gradient descent (gd) optimizer to find the minimax risk
#
# Author   : Mostafa Samir
# Written  : December 29, 2021
# Last Mod : December 29, 2021

from typing import List, Tuple, Union

import cvxpy as cp
import jax
import jax.numpy as jnp
import miniball as mb
import numpy as np
import optax
from scipy import stats

from uq4k.models.loss import DifferentaibleMeritFunc


class GdOpt:
    def __init__(self, objective: DifferentaibleMeritFunc) -> None:
        """initializes the gradient descent optimizer with a differentiable merit function

        Parameters
        ----------
        objective : DifferentaibleMeritFunc
            the merit function to optimize
        """
        self.objective = objective

    def compute_M_alpha(
        self,
        sigma_2: np.ndarray,
        mle_error: float,
        df: int,
        conf_level: float = 0.95,
        man_delta: Union[None, float] = None,
    ) -> float:
        """calculates the slack defining the ellipsoid from the likelihood ratio
        Supports chi-square method and manual setting

        Parameters
        ----------
        sigma_2 : np.ndarray
            Data variance
        mle_error : float
            MLE estimator's error
        df : int
            Degrees of freedom for the Chi-square distribution
        conf_level : float, optional
            desired confidence level (beta* in the paper), by default 0.95
        man_delta : Union[None, float], optional
            manual value of delta bypassing the chi-squqre method, by default None

        Returns
        -------
        float
            the value of M_alpha
        """
        if man_delta:
            delta = man_delta
        else:
            gamma = stats.chi2(df=df).ppf(conf_level)
            delta = gamma * sigma_2

        return mle_error + delta

    def find_mle(
        self, initial_theta: jax.numpy.DeviceArray, max_epoch: int = 1000, lr: float = 0.01
    ) -> Tuple[jax.numpy.DeviceArray, jax.numpy.DeviceArray]:
        """retrieves the MLE solution by applying gradient descent

        Parameters
        ----------
        initial_theta : jax.numpy.DeviceArray
            the initial value of theta to start GD from
        max_epoch : int, optional
            the number of epochs to run GD for, by default 1000
        lr : float, optional
            the learning rate, by default 0.01

        Returns
        -------
        Tuple[jax.numpy.DeviceArray, jax.numpy.DeviceArray]
            - MLE estimate of theta
            - value of MLE error
        """
        optimizer = optax.adam(learning_rate=lr)
        state = optimizer.init(initial_theta)
        theta = initial_theta
        error = None

        @jax.jit
        def update_step(theta, state):
            error, grad = jax.value_and_grad(self.objective.sum_sq_norms)(theta)
            updates, new_state = optimizer.update(grad, state)
            new_theta = optax.apply_updates(theta, updates)
            return error, new_theta, new_state

        for _ in range(max_epoch):
            error, theta, state = update_step(theta, state)

        return theta, error

    def optimize_min_e_ball(
        self,
        sigma_2: np.ndarray,
        data: np.ndarray,
        initial_theta: jax.numpy.DeviceArray,
        raduis_eps: float,
        conf_level: float,
        max_epoch: int = 100000,
        man_delta: Union[None, float] = None,
        bounds=None,
    ) -> Tuple[np.ndarray, float, List[np.ndarray], np.ndarray, float]:
        """runs the UQ4K optimization problem to find the minimum enclosing ball

        Parameters
        ----------
        sigma_2 : np.ndarray
            data varainace
        data : np.ndarray
            the data array
        initial_theta : jax.numpy.DeviceArray
            the initial starting point for the GD optimization
        raduis_eps : float
            the stopping criterion for the min enclosing ball optimization
        conf_level : float
            confidence level usded in chi-squared calculation of M_alpha
        max_epoch : int, optional
            the maximum number of epochs to run GD for, by default 100000
        man_delta : Union[None, float], optional
            manual value of delta bypassing the chi-squared method of M_alpha, by default None
        bounds : [type], optional
            the bounds for the theta vector (currently not used in GD), by default None

        Returns
        -------
        Tuple[np.ndarray, float, List[np.ndarray], np.ndarray, float]
            - mle_theta: MLE estimator of the theta given data
            - M_alpha: the level set constraint
            - S: collection of points for the minimum enclosing ball
            - center: the center of the minimum enclosing ball
            - raduis: the raduis of the minimum enclosing ball
        """
        pass
