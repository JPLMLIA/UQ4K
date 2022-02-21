"""this module contains an implementation of early stopping strategy
"""
from typing import Tuple

import numpy as np


class EarlyStopper:
    """implements early stopping strategy"""

    def __init__(
        self,
        min_improvement: float,
        patience: int,
        improvement_direction: str = 'down',
        improvement_type: str = 'absolute',
    ):
        """constructs the early stopper object

        Parameters
        ----------
        min_improvement : float
            minimum change in metric to qualify as an improvement
        patience : int
            number of epochs to wait for improvement before stopping
        improvement_direction: str
            if 'down', then the lower the metric, the better
            if 'up', then the higher the metric, the better
        improvement_type: str
            if 'absolute', then compare the absolute difference between the metric and its current best value
            if 'relative', then divide the absolute difference by the best metric value
        """
        self.min_improvement = min_improvement
        self.patience = patience
        self.direction_multiplier = 1 if improvement_direction == "up" else -1
        self.improvement_type = improvement_type

        self.best_metric = None
        self.best_epoch = None
        self.waiting = 0

    def check(self, metric: float, epoch_number: int) -> Tuple[bool, bool]:
        """checks if we should stop given the given metric

        Parameters
        ----------
        metric : float
            the metric value
        epoch_number: int
            the epoch number assotatied with the metric

        Returns
        -------
        Tuple[bool, bool]
            - whether an improvement happened
            - whether to stop training
        """
        metric = np.asarray(metric)

        stop = False
        improvement = False

        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch_number
        else:
            difference = self.direction_multiplier * (metric - self.best_metric)
            if self.improvement_type == 'relative':
                difference /= np.abs(self.best_metric)

            if difference >= self.min_improvement:
                self.best_metric = metric
                self.best_epoch = epoch_number
                self.waiting = 1
                improvement = True
            else:
                if self.waiting >= self.patience:
                    stop = True
                self.waiting += 1

        return improvement, stop
