import numpy as np
from typing import Union
from collections.abc import Callable
class Loss:
    def __init__(
        self,
        method = "lmse",
        **kwargs
    ):
        
        self.method = self._get_fun(method)
        self.method_args = kwargs
    
    def _get_fun(self, method):
        if callable(method):
            return method
        if method == "lmse":
            return lmse
        if method == "mse":
            return mse
        
    def calculate_loss(
        self,
        formants, 
        smoothed
    ):
        return self.method(formants, smoothed, **self.method_args)
    
def lmse(formants, smoothed, axis = 1):
    """
    calculate the log mean squared error
    """
    sqe = np.power(np.log(formants) - np.log(smoothed), 2)
    mse = np.nanmean(sqe, axis = axis)
    return mse

def mse(formants, smoothed, axis = 1):
    """
    calculate the mean squared error
    """
    sqe = np.power(formants - smoothed, 2)
    mse = np.nanmean(sqe, axis = axis)
    return mse