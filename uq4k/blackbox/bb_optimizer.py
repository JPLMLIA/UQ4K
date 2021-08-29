# Blackbox (bb) optimizer to find the minimax risk
#
# Author   : Mike Stanley
# Written  : August 28, 2021
# Last Mod : August 28, 2021

import cvxpy as cp
import miniball as mb
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy import stats


class BbOpt:

    def __init__(self, objective_obj, forward_model):
        """
        Parameters:
        -----------
            objective_obj (Objective) : contains objective function for opt
            forward_model (BaseModel) : forward operator -> objective func
        """
        self.objective_obj = objective_obj
        self.forward_model = forward_model

    def find_mle(self, data, starting_theta, max_iters=10):
        """ Find maximum likelihood estimator """
        coverged = False
        i = 0

        while (not coverged) & (i < max_iters):

            opt_mle = minimize(
                fun=lambda theta: self.objective_obj.sum_sq_norms(
                    data=data,
                    param=theta
                ),
                x0=starting_theta
            )

            if opt_mle['success']:
                coverged = True

            i += 1

        return opt_mle['x'], opt_mle['fun']

    def compute_M_alpha(
        self,
        sigma_2,
        mle_error,
        df,
        conf_level=0.95,
        man_delta=None
    ):
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
        epsilon_0=0.0001,
        conf_lev=0.95,
        man_delta=None,
        mu=1e3,
        bounds=[[-10, 10]]*2,
        max_iter=10
    ):
        """
        Primary data objects:
        - S         : set of optimized points
        - epsilon_0 : stopping criterion
        - beta      : significance level

        TODO: Add the QoI capability. Currently only works with identity map.

        Parameters:
            sigma_2   (float)  : data variance
            data      (np arr) : (n,) data array
            epsilon_0 (float)  : stopping criterion
            conf_lev  (float)  : confidence level used in chi-sq calc of delta
            man_delta (float)  : man. set delta (default None uses chi-sq calc)
            mu        (float)  : penalty coefficient for merit function
            bounds    (list)   : list bounds for use in the diff evolution algo
            max_iter  (int)    : max # of iterations of finding boundary points

        Returns:
            mle_theta (np arr) : MLE of parameters given data
            M_delta   (float)  : level-set constraint
            S         (list)   : collection of points for find min enclosing
                                 ball
            center    (np arr) : converged center
            radius_0  (float)  : converged radius of minimum enclosing ball
        """
        S = []
        d = len(bounds)

        # find the MLE
        mle_theta, mle_error = self.find_mle(data=data)

        # compute M_delta
        M_delta = self.compute_M_alpha(
            sigma_2=sigma_2,
            mle_error=mle_error,
            df=len(mle_theta),
            man_delta=man_delta,
            conf_level=conf_lev
        )

        # set variables for starting loop
        center = mle_theta.copy()
        S.append(center)
        radius_0 = 0
        e = 2 * epsilon_0
        i = 0

        while (e >= epsilon_0) & (i < max_iter):

            # find boundary point
            de_result = differential_evolution(
                func=self.objective_obj,
                args=(center),
                bounds=bounds
            )
            assert de_result['success']

            # check if new point has larger radius
            if np.linalg.norm(de_result['x'] - center) >= radius_0:
                S.append(de_result['x'])

            # find the minimum enclosing ball for S
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

        return mle_theta, M_delta, np.array(S), center, radius_0

    def weight_optimization(self, S):
        """
        Find dirac weights after min enclosing ball opt

        Parameters:
            S (np arr) : n x m, n - num diracs | m - dim each dirac

        Returns:
            optimized weights over diracs (n,) numpy array
        """
        # find the optimization objects
        n, m = S.shape
        Q_mat = np.zeros(shape=(n, n))

        for t in range(m):
            Q_mat += np.outer(S[:, t], S[:, t])

        v = np.square(S).sum(axis=1)

        # perform the optimization
        p_vec = cp.Variable(n)

        problem = cp.Problem(
            objective=cp.Minimize(
                cp.quad_form(p_vec, Q_mat) - v.T @ p_vec
            ),
            constraints=[
                p_vec >= np.zeros(n),
                cp.sum(p_vec) == 1
            ]
        )

        # solve and check convergence
        problem.solve()
        assert problem.status == 'optimal'

        return p_vec.value
