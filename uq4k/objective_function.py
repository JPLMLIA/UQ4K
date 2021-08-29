# Implements objective function class for UQ4K. Provides a general objective
# function framework, which can take an arbitrary forward model.
#
# Author   : Mike Stanley
# Written  : August 26, 2021
# Last Mod : August 26, 2021

from abc import ABC, abstractmethod
import numpy as np


class Objective(ABC):  # TODO: should these abstract methods be defined here?

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sum_sq_norms(self):
        """
        Finds the squared 2-norm of the difference between model and data
        """
        pass

    @abstractmethod
    def center_dist(self):
        """
        Finds the squared 2-norm between a new proposed parameter value and
        the current center
        """
        pass


class MeritFunc(Objective):
    def __init__(self, forward_model, mu, data):
        """
        Dimension key:
            n : number of data points
            d : dimension of each data point

        Parameters:
        -----------
            forward_model (BaseModel) : see base_model.py
            mu            (float)     : merit function parameter
            data          (np arr)    : array of observed data - n x d
        """
        self.forward_model = forward_model
        self.mu = mu
        self.data = data

    def sum_sq_norms(self, params):
        """
        Finds the squared 2-norm of the difference between model and data

        Dimension key:
            p : dimension of model parameters

        Parameters:
        -----------
            params (np arr) : p

        Returns:
        --------
            2-norm of residuals
        """
        diffs = self.data - self.forward_model(params)
        diff1_sq = np.square(diffs[:, 0])
        diff2_sq = np.square(diffs[:, 1])
        return (diff1_sq + diff2_sq).sum()

    def center_dist(self, new_point, center):
        """
        Finds the squared 2-norm between a new proposed parameter value and
        the current center

        Dimension key:
            p : dimension of model parameters

        Parameters:
        -----------
            new_point (np arr) : p
            center    (np arr) : p

        Returns:
        --------
            squared 2-norm of distance between two points
        """
        return np.linalg.norm(new_point - center) ** 2

    def __call__(self, new_point, center, M_alpha):
        """
        Evaluates the objective function at some new point.

        Dimension key:
            p : dimension of model parameters

        Parameters:
        -----------
            new_point (np arr) : p
            center    (np arr) : p
            M_alpha   (float)  : bound on the error

        Returns:
        --------
            Objective function
        """
        # find the distance from center
        center_dist_term = self.center_dist(
            new_point=new_point,
            center=center
        )

        # compute the penalty term
        error = self.sum_sq_norms(params=new_point)
        merit_term = self.mu * np.max(np.array([0, error - M_alpha]))

        return - center_dist_term + merit_term
