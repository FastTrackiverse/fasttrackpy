import numpy as np
from typing import Union
from collections.abc import Callable
class Loss:
    """_summary_

    Args:
        method (Union[str, Callable], optional): _description_. Defaults to "lmse".
    """

    def __init__(
        self,
        method: Union[str, Callable] = "lmse",
        **kwargs
    ):
        self.method = self._get_fun(method)
        self.method_args = kwargs
    
    def _get_fun(
            self, 
            method: Union[str, Callable]
        ) -> Callable:
        if callable(method):
            return method
        if method == "lmse":
            return lmse
        if method == "mse":
            return mse
        
    def calculate_loss(
        self,
        formants: np.ndarray, 
        smoothed: np.ndarray
    ):
        return self.method(formants, smoothed, **self.method_args)
    
def lmse(
        formants: np.ndarray, 
        smoothed: np.ndarray, 
        axis: int = 1
    ) -> np.ndarray:
    """_summary_

    Args:
        formants (np.ndarray): _description_
        smoothed (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    sqe = np.power(np.log(formants) - np.log(smoothed), 2)
    mse = np.nanmean(sqe, axis = axis)
    return mse

def mse(
        formants: np.ndarray, 
        smoothed: np.ndarray, 
        axis: int = 1
    ) -> np.ndarray:
    """_summary_

    Args:
        formants (np.ndarray): _description_
        smoothed (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    sqe = np.power(formants - smoothed, 2)
    mse = np.nanmean(sqe, axis = axis)
    return mse