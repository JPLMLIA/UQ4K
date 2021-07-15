import pathlib
import sys
from collections import Iterable

import numpy as np

# set the paths #TODO: fix this later
path = pathlib.Path.cwd().parent.parent
sys.path.insert(0, str(path))

# sklearn module
from sklearn.linear_model import LinearRegression

# uq4k modules
from uq4k.models.base_model import BaseModel, Modelparameter


class LinearModel(BaseModel):
    """function model for testing the uq pipelines with weight and bias
    in a linear model.

    This gives an example on how one can build any new ML or
    physics based (pde) models to perform DTUQ.


    Parameters
    ----------
    w : float
        the weight of the model.
    b : float
        the bias of the model.

    Examples
    --------
    >>> from models import LinearModel
    >>> model = LinearModel()
    >>> model.get_params()
    """

    def __init__(self, weight=1.0, weight_bounds=(1e-5, 1e5), bias=1.0, bias_bounds=(1e-5, 1e5)):
        self.weight = weight
        self.weight_bounds = weight_bounds
        self.bias = bias
        self.bias_bounds = bias_bounds
        self.model = LinearRegression()
        if isinstance(self.weight, Iterable):
            self.model.fit(np.ones((1, len(self.weight))), np.ones((1, 1)))
        else:
            self.model.fit(np.ones((1, 1)), np.ones((1, 1)))

    @property
    def modelparameter_weight(self):

        if isinstance(self.weight, Iterable):
            return Modelparameter(
                "weight",
                "numeric",
                self.weight_bounds,
                len(self.weight),
            )
        else:
            return Modelparameter("weight", "numeric", self.weight_bounds)

    @property
    def modelparameter_bias(self):

        if isinstance(self.bias, Iterable):
            return Modelparameter(
                "bias",
                "numeric",
                self.bias_bounds,
                len(self.bias),
            )
        else:
            return Modelparameter("bias", "numeric", self.bias_bounds)

    def _check_input(self, X):

        if np.atleast_2d(X).shape[1] != np.atleast_2d(self.weight).shape[1]:
            raise ValueError(" input dimension does not match the models dimension !")

    def __call__(self, X):
        """Return the model value by invoking the model class.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)


        Returns
        -------
        y : ndarray of shape (n_samples_X, n_target)

        """
        # check the validity of the X
        self._check_input(X)

        # update the weights and biases of the model
        self.weight = self.get_params().get("weight")
        self.bias = self.get_params().get("bias")
        self.model.coef_ = np.atleast_2d(self.weight)
        self.model.intercept_ = np.atleast_2d(self.bias)

        return self.model.predict(X)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":

    model = LinearModel()
    X = np.array([1, 2, 3]).reshape(-1, 1)
    print(model(X))
    print(model.theta)
