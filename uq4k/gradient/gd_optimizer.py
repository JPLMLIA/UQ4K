# gradient descent (gd) optimizer to find the minimax risk
#
# Author   : Mostafa Samir
# Written  : December 29, 2021
# Last Mod : December 29, 2021

import random
from functools import partial
from typing import List, Tuple, Union

import cvxpy as cp
import jax
import jax.numpy as jnp
import miniball as mb
import numpy as np
import optax
from scipy import stats

from uq4k.gradient.early_stopper import EarlyStopper
from uq4k.models.loss import DifferentaibleMeritFunc


class GdOpt:
    @staticmethod
    def __multisample_startegy(
        dims: int, num_samples: int = 10, bounds: Tuple[int, int] = (-1, 1), seed: int = 0
    ) -> jax.numpy.DeviceArray:
        """generates multiple random initializations of the theta parameter

        Parameters
        ----------
        dims: int
            the dimesnions of the theta parameters
        num_samples : int, optional
            the number of samples to draw, by default 10
        bounds: Tuple[int, int], optional
            the lower and upper bound to sample uniformly from
        seed: int, optional
            the seed to the random number generator, by default 0

        Returns
        -------
        jax.numpy.DeviceArray
            the sample of random initializations (num_samples, parameters_count)
        """
        rng = jax.random.PRNGKey(seed)
        lower_bound, upper_bound = bounds

        candidates = jax.random.uniform(
            rng, shape=(num_samples, dims), minval=lower_bound, maxval=upper_bound
        )

        return candidates

    def __init__(self, objective: DifferentaibleMeritFunc) -> None:
        """initializes the gradient descent optimizer with a differentiable merit function

        Parameters
        ----------
        objective : DifferentaibleMeritFunc
            the merit function to optimize
        """
        self.objective = objective
        self.init_strategies = {"multisample": GdOpt.__multisample_startegy}

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
        self,
        theta_dims: int,
        max_epoch: int = 1000,
        lr: float = 0.01,
        min_improvement: float = 0.01,
        patience: int = 1000,
    ) -> Tuple[jax.numpy.DeviceArray, jax.numpy.DeviceArray]:
        """retrieves the MLE solution by applying gradient descent

        Parameters
        ----------
        theta_dims : int
            diminsionality of the theta paramter
        max_epoch : int, optional
            the number of epochs to run GD for, by default 1000
        lr : float, optional
            the learning rate, by default 0.01
        min_improvement: float, optional
            the relative minimum difference between loss to consider an improvement,
            default to 0.1 (10% lower than the smallest observed loss)
        patiance: int, optional
            the number of epochs to wait for improvement in the loss before stopping


        Returns
        -------
        Tuple[jax.numpy.DeviceArray, jax.numpy.DeviceArray]
            - MLE estimate of theta
            - value of MLE error
        """
        early_stopper = EarlyStopper(
            min_improvement=min_improvement, patience=patience, improvement_type="relative"
        )
        rng = jax.random.PRNGKey(seed=42)
        init_theta = jax.random.uniform(rng, shape=(theta_dims,), minval=-1, maxval=1)
        optimizer = optax.adam(learning_rate=lr)
        state = optimizer.init(init_theta)
        theta = init_theta
        error = None

        @jax.jit
        def update_step(theta, state):
            error, grad = jax.value_and_grad(self.objective.sum_sq_norms)(theta)
            normed_grad = grad / jnp.linalg.norm(grad)
            updates, new_state = optimizer.update(normed_grad, state)
            new_theta = optax.apply_updates(theta, updates)
            return error, new_theta, new_state

        for i in range(max_epoch):
            error, theta, state = update_step(theta, state)
            _, stop = early_stopper.check(error, i)
            if stop:
                break

        return theta, error

    def __get_furthest_point(
        self,
        initial_theta: jax.numpy.DeviceArray,
        M_alpha: float,
        center: np.ndarray,
        max_epoch: int,
        lr: float,
        min_improvement: float = 0.01,
        patience: int = 1000,
    ) -> jax.numpy.DeviceArray:
        """performs gradient descent on the whole objective function to get the furthest point from the center

        Parameters
        ----------
        initial_theta: jax.numpy.DeviceArray
            the initial value of theta to start from
        M_alpha : float
            the slack defining the ellipsoid from the likelihood ratio
        center : np.ndarray
            the center to find the furthest point from
        max_epoch : int
            max number of epoch to run the optimization for
        lr : float
            the learning rate used
        min_improvement: float, optional
            the relative minimum difference between loss to consider an improvement,
            default to 0.1 (10% lower than the smallest observed loss)
        patiance: int, optional
            the number of epochs to wait for improvement in the loss before stopping

        Returns
        -------
        jax.numpy.DeviceArray
            the furthest point found
        """
        early_stopper = EarlyStopper(
            min_improvement=min_improvement, patience=patience, improvement_type='relative'
        )
        optimizer = optax.adam(learning_rate=lr)
        state = optimizer.init(initial_theta)
        theta = initial_theta

        bound_objective = partial(self.objective, center=center, M_alpha=M_alpha)

        furthest_point = None
        furthest_distance = 0

        @jax.jit
        def update_step(theta, state):
            loss, grad = jax.value_and_grad(bound_objective)(theta)
            normed_grad = grad / jnp.linalg.norm(grad)
            updates, new_state = optimizer.update(normed_grad, state)
            new_theta = optax.apply_updates(theta, updates)
            return loss, new_theta, new_state

        for i in range(max_epoch):
            loss, theta, state = update_step(theta, state)
            improvement, stop = early_stopper.check(loss, i)
            if stop:
                break
            if improvement:
                furthest_distance = jax.lax.stop_gradient(self.objective.center_dist(theta, center))
                furthest_point = jax.lax.stop_gradient(self.objective.qoi_func(theta))

        return furthest_point, furthest_distance

    def optimize_min_e_ball(
        self,
        sigma_2: np.ndarray,
        data: np.ndarray,
        initial_theta: Union[jax.numpy.DeviceArray, str],
        theta_dims: int,
        raduis_eps: float,
        conf_level: float,
        max_epoch: int = 100000,
        lr: float = 0.001,
        man_delta: Union[None, float] = None,
        bounds=None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, float, List[np.ndarray], np.ndarray, float]:
        """runs the UQ4K optimization problem to find the minimum enclosing ball

        Parameters
        ----------
        sigma_2 : np.ndarray
            data varainace
        data : np.ndarray
            the data array
        initial_theta : Union[jax.numpy.DeviceArray, str]
            array: the initial starting point for the GD optimization
            str: the initialization startegy used internally
                - 'multisample': sample multiple random initializations
                   and continue with the one yielding the furthest distance
        theta_dims: int
            the dimensionality of the theta parameters
        raduis_eps : float
            the stopping criterion for the min enclosing ball optimization
        conf_level : float
            confidence level usded in chi-squared calculation of M_alpha
        max_epoch : int, optional
            the maximum number of epochs to run GD for, by default 100000
        lr: float, optional
            the learning rate to be used
        man_delta : Union[None, float], optional
            manual value of delta bypassing the chi-squared method of M_alpha, by default None
        bounds : [type], optional
            the bounds for the theta vector (currently not used in GD), by default None
        seed: int, optional
            random seed for theta initialization

        Returns
        -------
        Tuple[np.ndarray, float, List[np.ndarray], np.ndarray, float]
            - mle_theta: MLE estimator of the theta given data
            - M_alpha: the level set constraint
            - S: collection of points for the minimum enclosing ball
            - center: the center of the minimum enclosing ball
            - raduis: the raduis of the minimum enclosing ball
        """
        dims = theta_dims
        qoi_dims = self.objective.qoi_func(np.zeros(shape=(dims,))).size

        initial_theta_val = None
        init_strategy = None
        if isinstance(initial_theta, str):
            init_strategy = self.init_strategies.get(initial_theta, None)
            if init_strategy is None:
                raise KeyError(f"No initialization strategy called {initial_theta}")
        else:
            initial_theta_val = jnp.reshape(initial_theta, (1, -1))

        S = []
        center = None
        raduis = 0

        mle_theta, mle_error = self.find_mle(theta_dims, max_epoch, lr)

        M_alpha = self.compute_M_alpha(
            sigma_2, mle_error, df=dims, conf_level=conf_level, man_delta=man_delta
        )

        center = np.asarray(jax.lax.stop_gradient(self.objective.qoi_func(mle_theta)))
        S.append(center)

        raduis_diff = np.inf

        while raduis_diff > raduis_eps:
            seed += 2

            init_theta_candidates = (
                init_strategy(dims=theta_dims, seed=seed) if init_strategy else initial_theta_val
            )

            furthest_point = None
            furthest_distance = 0
            for init_theta in init_theta_candidates:
                furthest_point_candidate, furthest_distance_candidate = self.__get_furthest_point(
                    init_theta, M_alpha, center, max_epoch, lr
                )
                if furthest_distance_candidate >= furthest_distance:
                    furthest_distance = furthest_distance_candidate
                    furthest_point = furthest_point_candidate
            furthest_point = np.asarray(furthest_point)

            S.append(furthest_point)
            S_array = np.array(S)
            if S_array.ndim == 1:
                center, r_squared = mb.get_bounding_ball(S_array[:, np.newaxis])
            else:
                center, r_squared = mb.get_bounding_ball(S_array)

            raduis_diff = np.abs(np.sqrt(r_squared) - raduis)
            raduis = np.sqrt(r_squared)

            if len(S) > qoi_dims + 1:
                distances = [np.linalg.norm(center - Si) for Si in S]
                remove_indx = np.argmin(distances)
                S.pop(remove_indx)

        return mle_theta, M_alpha, np.array(S), center, raduis

    def weight_optimization(self, S):
        """
        Find dirac weights after min enclosing ball opt

        Parameters:
            S (np arr) : n x m, n - num diracs | m - dim each dirac

        Returns:
            optimized weights over diracs (n,) numpy array
        """
        # find the optimization objects
        ONE_D = len(S.shape) == 1
        if ONE_D:
            n = S.shape[0]

        else:
            n, m = S.shape

        Q_mat = np.zeros(shape=(n, n))

        if ONE_D:
            Q_mat = np.outer(S, S)
            v = np.square(S)

        else:
            for t in range(m):
                Q_mat += np.outer(S[:, t], S[:, t])

            v = np.square(S).sum(axis=1)

        # perform the optimization
        p_vec = cp.Variable(n)

        problem = cp.Problem(
            objective=cp.Minimize(cp.quad_form(p_vec, Q_mat) - v.T @ p_vec),
            constraints=[p_vec >= np.zeros(n), cp.sum(p_vec) == 1],
        )

        # solve and check convergence
        problem.solve()
        assert problem.status == 'optimal'

        return p_vec.value
