"""
this is the main base model for the uq4k
"""
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature

import numpy as np


class Modelparameter(namedtuple("Modelparameters", ("name", "value_type", "bounds", "n_elements", "fixed"))):
    """define the parameters of the model


    Attributes
    ----------
    name : str
        The name of the Modelparameter. Note that a model using a
        Modelparameter with name "x" must have the attributes self.x and
        self.x_bounds

    value_type : str
        The type of the Modelparameter. Currently, only "numeric"
        Modelparameters are supported.

    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the Modelparameter's value
        cannot be changed.

    n_elements : int, default=1
        The number of elements of the Modelparameter value. Defaults to 1,
        which corresponds to a scalar Modelparameter. n_elements > 1
        corresponds to a Modelparameter which is vector-valued.

    fixed : bool, default=None
        Whether the value of this Modelparameter is fixed, i.e., cannot be
        changed during Modelparameter tuning. If None is passed, the "fixed" is
        derived based on the given bounds.

    Examples
    --------
    >>> #fill this part later
    """

    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None):
        if not isinstance(bounds, str) or bounds != "fixed":
            bounds = np.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError(
                        "Bounds on %s should have either 1 or "
                        "%d dimensions. Given are %d" % (name, n_elements, bounds.shape[0])
                    )

        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        return super(Modelparameter, cls).__new__(cls, name, value_type, bounds, n_elements, fixed)

    # check the equality of Modelparameters
    def __eq__(self, other):
        return (
            self.name == other.name
            and self.value_type == other.value_type
            and np.all(self.bounds == other.bounds)
            and self.fixed == other.fixed
        )


class BaseModel(metaclass=ABCMeta):
    """Base class for all models."""

    def get_params(self, deep=True):

        """Get parameters of this model.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = cls.__init__
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        for arg in args:
            params[arg] = getattr(self, arg)

        return params

    def set_params(self, **params):
        """Set the parameters of this model.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for model %s. "
                    "Check the list of available parameters "
                    "with `model.get_params().keys()`." % (key, self.__class__.__name__)
                )
            setattr(self, key, value)
        return self

    @property
    def n_dims(self):
        """Returns the number of non-fixed modelparameters of the model."""
        return self.theta.shape[0]

    @property
    def modelparameters(self):
        """Returns a list of all modelparameters specifications."""
        r = [getattr(self, attr) for attr in dir(self) if attr.startswith("modelparameter_")]
        return r

    @property
    def theta(self):
        """Returns the (flattened) non-fixed modelparameters.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, modelparameters of the model
        """
        theta = []
        params = self.get_params()
        for modelparameter in self.modelparameters:
            if not modelparameter.fixed:
                theta.append(params[modelparameter.name])
        if len(theta) > 0:
            return np.hstack(theta)
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened) non-fixed modelparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed,  modelparameters of the model
        """
        params = self.get_params()
        i = 0
        for modelparameter in self.modelparameters:
            if modelparameter.fixed:
                continue
            if modelparameter.n_elements > 1:
                # vector-valued parameter
                params[modelparameter.name] = theta[i : i + modelparameter.n_elements]
                i += modelparameter.n_elements
            else:
                params[modelparameter.name] = theta[i]
                i += 1

        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries." " Should be %d; given are %d" % (i, len(theta))
            )
        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the model's Modelparameters theta
        """
        bounds = [
            modelparameter.bounds for modelparameter in self.modelparameters if not modelparameter.fixed
        ]
        if len(bounds) > 0:
            return np.vstack(bounds)
        else:
            return np.array([])

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        params_a = self.get_params()
        params_b = other.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.theta)))

    @abstractmethod
    def __call__(self, X):
        """Evaluate the model."""

    def _check_bounds_params(self):
        """Called after fitting to warn if bounds may have been too tight."""
        list_close = np.isclose(self.bounds, np.atleast_2d(self.theta).T)
        idx = 0
        for hyp in self.modelparameters:
            if hyp.fixed:
                continue
            for dim in range(hyp.n_elements):
                if list_close[idx, 0]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified lower "
                        "bound %s. Decreasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][0])
                    )
                elif list_close[idx, 1]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified upper "
                        "bound %s. Increasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][1])
                    )
                idx += 1
