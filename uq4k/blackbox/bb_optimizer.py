# Blackbox (bb) optimizer to find the minimax risk
#
# Author   : Mike Stanley
# Written  : August 28, 2021
# Last Mod : November 20, 2021

import cvxpy as cp
import miniball as mb
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution, minimize


class BbOpt:
    def __init__(self, objective_obj):
        """
        Parameters:
        -----------
            objective_obj (Objective) : contains objective function for opt
        """
        self.objective_obj = objective_obj

    def find_mle(self, data, starting_theta, max_iters=10):
        """
        Find maximum likelihood estimator

        Parameters:
        -----------
            data (np arr)
        """
        coverged = False
        i = 0

        while (not coverged) & (i < max_iters):

            opt_mle = minimize(
                fun=lambda theta: self.objective_obj.sum_sq_norms(params=theta), x0=starting_theta
            )

            if opt_mle['success']:
                coverged = True

            i += 1

        return opt_mle['x'], opt_mle['fun']

    def compute_M_alpha(self, sigma_2, mle_error, df, conf_level=0.95, man_delta=None):
        """
        Finds the slack to define the ellipsoid from the likelihood ratio.
        Supports chi-sq method and manual delta set

        Use man_delta for debug.

        NOTE: "delta" is a shorthand for 2 sigma^2 ln(1 / alpha) as written
        in the paper.
        """
        if man_delta:
            delta = man_delta
        else:
            gamma = stats.chi2(df=df).ppf(conf_level)
            delta = gamma * sigma_2

        return mle_error + delta

    def optimize_min_e_ball(
        self,
        sigma_2,
        data,
        theta_init,
        epsilon_0,
        conf_lev,
        bounds,
        max_iter,
        man_delta=None,
    ):
        """
        Primary data objects:
        - S         : set of optimized points
        - epsilon_0 : stopping criterion
        - beta      : significance level

        Parameters:
            sigma_2    (float)  : data variance
            data       (np arr) : (n,) data array
            theta_init (np arr) : starting point for MLE optimization
            epsilon_0  (float)  : stopping criterion
            conf_lev   (float)  : confidence level used in chi-sq calc of delta
            bounds     (list)   : list bounds use in the diff evolution algo
            max_iter   (int)    : max # of iterations finding boundary points
            man_delta  (float)  : man. set delta (default None - chi-sq calc)

        Returns:
            mle_theta (np arr) : MLE of parameters given data
            M_alpha   (float)  : level-set constraint
            S         (list)   : collection of points for find min enclosing
                                 ball
            center    (np arr) : converged center
            radius_0  (float)  : converged radius of minimum enclosing ball
        """
        S = []
        d = len(bounds)

        # find the MLE
        mle_theta, mle_error = self.find_mle(data=data, starting_theta=theta_init)

        # compute M_alpha
        M_alpha = self.compute_M_alpha(
            sigma_2=sigma_2, mle_error=mle_error, df=len(mle_theta), man_delta=man_delta, conf_level=conf_lev
        )

        # set variables for starting loop
        center = self.objective_obj.qoi_func(mle_theta).copy()
        S.append(center)
        radius_0 = 0
        e = 2 * epsilon_0
        i = 0

        while (e >= epsilon_0) & (i < max_iter):

            # find boundary point
            de_result = differential_evolution(func=self.objective_obj, args=(center, M_alpha), bounds=bounds)
            assert de_result['success']

            # check if new point has larger radius
            if np.linalg.norm(de_result['x'] - center) >= radius_0:
                S.append(self.objective_obj.qoi_func(de_result['x']))

            # find the minimum enclosing ball for S
            if len(np.array(S).shape) == 1:  # i.e., a 1d QoI
                C, r2 = mb.get_bounding_ball(np.array(S)[:, np.newaxis])
                center = C[0]
            else:
                C, r2 = mb.get_bounding_ball(np.array(S))
                center = C

            # update radius change
            e = np.abs(np.sqrt(r2) - radius_0)
            radius_0 = np.sqrt(r2)

            # check size of set S -- potentially do not need this
            if len(S) > d + 1:

                # eliminate element in S with smallest distance from c
                distances = [np.linalg.norm(center - S_i) for S_i in S]
                remove_idx = np.argmin(distances)
                del S[remove_idx]

            i += 1

        return mle_theta, M_alpha, np.array(S), center, radius_0

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
