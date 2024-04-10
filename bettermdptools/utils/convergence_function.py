import functools
from typing import Callable

import numpy as np

"""
This function should take in the past and current values of the utility function and return a boolean indicating if you
have converged.  This is useful for iterative algorithms like value iteration and policy iteration.
"""
UtilityConvergenceFunction = Callable[[np.ndarray, np.ndarray], bool]

def get_max_value_less_than_theta(theta: float) -> UtilityConvergenceFunction:
    """
    This returns a convergence function that requires the maximum difference between the past and current utility is
    less than theta.

    Args:
        theta (float): The value that the maximum difference between the past and current utility must be less than.

    Returns:
        UtilityConvergenceFunction: A function that takes in the past and current utility values and returns a boolean
            indicating if the maximum difference is less than theta.
    """
    return functools.partial(max_difference_less_than_theta, theta=theta)

def get_mean_value_less_than_theta(theta: float) -> UtilityConvergenceFunction:
    """
    This returns a convergence function that requires the mean difference between the past and current utility is less
    than theta.

    Args:
        theta (float): The value that the mean difference between the past and current utility must be less than.

    Returns:
        UtilityConvergenceFunction: A function that takes in the past and current utility values and returns a boolean
            indicating if the mean difference is less than theta.
    """
    return functools.partial(mean_difference_less_than_theta, theta=theta)

def max_difference_less_than_theta(past: np.ndarray, current: np.ndarray, theta: float) -> bool:
    return np.max(np.abs(past - current)) < theta

def mean_difference_less_than_theta(past: np.ndarray, current: np.ndarray, theta: float) -> bool:
    return np.mean(np.abs(past - current)) < theta
