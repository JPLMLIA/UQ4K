# Implements objective function class for UQ4K. Provides a general objective
# function framework, which can take an arbitrary forward model.
#
# Author   : Mike Stanley
# Written  : August 26, 2021
# Last Mod : November 20, 2021

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np


class AbstractLoss(ABC):  # TODO: should these abstract methods be defined here?
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


class MeritFunc(AbstractLoss):
    def __init__(self, forward_model, mu, data, qoi_func):
        """
        Dimension key:
            n : number of data points
            d : dimension of each data point
            m : dimension of the qoi

        Parameters:
        -----------
            forward_model (BaseModel) : see base_model.py
            mu            (float)     : merit function parameter
            data          (np arr)    : array of observed data - n x d
            qoi_func      (function)  : maps theta |-> qoi, R^n -> R^m

        """
        self.forward_model = forward_model
        self.mu = mu
        self.data = data
        self.qoi_func = qoi_func

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
        return np.square(diffs).sum()

    def center_dist(self, new_point, center):
        """
        Finds the squared 2-norm between a new proposed parameter value and
        the current center

        Dimension key:
            p : dimension of model parameters

        Parameters:
        -----------
            new_point (np arr) : p
            center    (np arr) : m

        Returns:
        --------
            squared 2-norm of distance between two points
        """
        return np.linalg.norm(self.qoi_func(new_point) - center) ** 2

    def __call__(self, new_point, center, M_alpha):
        """
        Evaluates the objective function at some new point.

        Dimension key:
            p : dimension of model parameters
            m : dimension of the QoI

        Parameters:
        -----------
            new_point (np arr) : p
            center    (np arr) : m
            M_alpha   (float)  : bound on the error

        Returns:
        --------
            Objective function
        """
        # find the distance from center
        center_dist_term = self.center_dist(new_point=new_point, center=center)

        # compute the penalty term
        error = self.sum_sq_norms(params=new_point)
        merit_term = self.mu * np.max(np.array([0, error - M_alpha]))

        return -center_dist_term + merit_term


class DifferentaibleMeritFunc(AbstractLoss):
    def __init__(self, forward_model, mu, data, qoi_func):
        """
        Dimension key:
            n : number of data points
            d : dimension of each data point
            m : dimension of the qoi

        Parameters:
        -----------
            forward_model (BaseModel) : see base_model.py
            mu            (float)     : merit function parameter
            data          (np arr)    : array of observed data - n x d
            qoi_func      (function)  : maps theta |-> qoi, R^n -> R^m
        """
        self.forward_model = forward_model
        self.mu = mu
        self.data = data
        self.qoi_func = qoi_func

    def sum_sq_norms(self, params):
        """
        Finds the squared 2-norm of the difference between model and data

        Dimension key:
            p : dimension of model parameters

        Parameters:
        -----------
            params (jax DeviceArray) : p

        Returns:
        --------
            2-norm of residuals
        """
        diffs_squared = jnp.square(self.data - self.forward_model(params))
        return jnp.sum(diffs_squared)

    def center_dist(self, new_point, center):
        """
        Finds the squared 2-norm between a new proposed parameter value and
        the current center

        Dimension key:
            p : dimension of model parameters

        Parameters:
        -----------
            new_point (jax DeviceArray) : p
            center    (jax DeviceArray) : m

        Returns:
        --------
            squared 2-norm of distance between two points
        """
        diffs_squared = jnp.square(self.qoi_func(new_point) - center)
        return jnp.sum(diffs_squared)

    def __call__(self, new_point, center, M_Alpha):
        """
        Evaluates the objective function at some new point.

        Dimension key:
            p : dimension of model parameters
            m : dimension of the QoI

        Parameters:
        -----------
            new_point (jax.numpy.DeviceArray) : p
            center    (np arr) : m
            M_alpha   (float)  : bound on the error

        Returns:
        --------
            Objective function
        """

        center_dist_term = self.center_dist(new_point, center)
        error = self.sum_sq_norms(params=new_point)
        constraint = self.mu * jnp.max([error - M_Alpha, 0])

        return -center_dist_term + constraint


class MeritFunc_NEW(AbstractLoss):
    def __init__(self, forward_model, mu, data_y, data_x):
        """
        Dimension key:
            n  : number of data points
            dx : dimension of each input
            dy : dimension of each output

        Parameters:
        -----------
            forward_model (BaseModel) : see base_model.py
            mu            (float)     : merit function parameter
            data_y        (np arr)    : array of observed output - n x dy
            data_x        (np arr)    : array of observed input - n x dx
        """
        self.forward_model = forward_model
        self.mu = mu
        self.data_y = data_y
        self.data_x = data_x

    def sum_sq_norms(self):
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
        diffs = self.data_y - self.forward_model(self.data_x)
        return np.square(diffs).sum()

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
        center_dist_term = self.center_dist(new_point=new_point, center=center)

        # compute the penalty term
        error = self.sum_sq_norms(params=new_point)
        merit_term = self.mu * np.max(np.array([0, error - M_alpha]))

        return -center_dist_term + merit_term
