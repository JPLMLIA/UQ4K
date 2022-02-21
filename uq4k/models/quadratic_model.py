# Quadratic Model (in x) from the UQ4K paper
#
# Author        : Mike Stanley
# Created       : Sep 30, 2021
# Last Modified : Sep 30, 2021

from collections.abc import Iterable

import numpy as np

from uq4k.models.base_model import BaseModel, Modelparameter


class QuadraticModel(BaseModel):
    """
    Implementation of the Quadratic (in X) model

    Key:
    - d = number of model parameters
    - n = number of data points

    Parameters:
    -----------
        theta        (np arr) : model parameters (d)
        theta_bounds (np arr) : bounds on model parameters (d x 2)
        x            (np arr) : observed x locations
    """

    def __init__(self, weight, weight_bounds):
        self.weight = weight
        self.weight_bounds = weight_bounds

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

    def __call__(self, X):
        """
        Evaluates the forward model for a vector of inputs

        Parameters:
        -----------
            X (np arr) : input values (N)

        Returns:
        --------
            forward output of data values
        """
        N = X.shape[0]
        D = self.weight.shape[0]

        powers = np.tile(np.arange(D)[:, np.newaxis], N).T
        X = np.power(np.tile(X[:, np.newaxis], D), powers)

        return X @ self.weight
