# The Lotka-Volterra model forward operator

# See 'https://en.wikipedia.org/wiki/Lotka–Volterra_equations' for the
# definitions of the coeffecients.

from base_model import BaseModel, Modelparameter
from collections import Iterable
from scipy.integrate import odeint


class LotkaVolterraModel(object):
    def __init__(self, y0=None):
        self._y0 = y0

    def simulate(self, parameters, times):
        """
        Simulates model dynamics for given model parameters and some
        number of time steps.

        Parameters:
            parameters (iterable) : list of model parameters in the order
                                    - alpha, beta, gamma, delta, Xt0, Yt0
            times      (list)     : sequence of time points
        """
        alpha, beta, gamma, delta, Xt0, Yt0 = [x for x in parameters]

        def rhs(y, t, p):
            X, Y = y
            dX_dt = alpha*X - beta*X*Y
            dY_dt = -gamma*Y + delta*X*Y
            return dX_dt, dY_dt

        values = odeint(rhs, [Xt0, Yt0], times, (parameters,))
        return values


class PredPrey(BaseModel):
    """
    Basic implementation of predator/prey model

    Parameters:
    -----------
        alpha        (float) : model parameter
        alpha_bounds (tup)   : bounds for alpha parameter
        gamma        (float) : model parameter
        gamma_bounds (tup)   : bounds for alpha parameter
        beta         (float) : model parameter
        delta        (float) : model parameter
        prey_init    (float) : model parameter
        pred_init    (float) : model parameter

    """
    def __init__(
        self,
        alpha,
        alpha_bounds,
        gamma,
        gamma_bounds,
        beta,
        delta,
        prey_init,
        pred_init
    ):

        # model parameter attributes -- only accepts parameters optimizing for
        self.alpha = alpha
        self.alpha_bounds = alpha_bounds
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

        self.beta = beta
        self.delta = delta
        self.prey_init = prey_init
        self.pred_init = pred_init

        # set solver
        self.ode_solver = LotkaVolterraModel()

    @property
    def modelparameter_alpha(self):
        """"""
        if isinstance(self.alpha, Iterable):
            return Modelparameter(
                "alpha",
                "numeric",
                self.alpha_bounds,
                len(self.alpha)
            )
        else:
            return Modelparameter(
                "alpha",
                "numeric",
                self.alpha_bounds
            )

    @property
    def modelparameter_gamma(self):
        """"""
        if isinstance(self.gamma, Iterable):
            return Modelparameter(
                "gamma",
                "numeric",
                self.gamma_bounds,
                len(self.gamma)
            )
        else:
            return Modelparameter(
                "gamma",
                "numeric",
                self.gamma_bounds
            )
