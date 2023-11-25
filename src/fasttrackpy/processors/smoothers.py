import numpy as np
import scipy.fft
from typing import Union
from collections.abc import Callable


class Smoothed:
    """Smooothed formant tracks
    Args:
        smoothed (np.ndarray): a (formants, time) shaped numpy array of 
            smoothed formant values
        params (np.ndarray, optional): Parameters (if any) of the smoother. 
            Defaults to None.
    """

    def __init__(
        self,
        smoothed: np.ndarray,
        params: np.ndarray = None
    ):
        self.smoothed = smoothed
        self.params = params

class Smoother:
    """A smoother function factory

    Args:
        method (Union[str, Callable], optional): The smoothing method to use.
            Defaults to "dct_smooth".
            Can be a custom smoother such that it takes a 1D array as input
            and returns a `Smoothed` class.
        kwargs : Any additional arguments or parameters for the `method`.
    """
    def __init__(
        self,
        method: Union[str, Callable] = "dct_smooth_regression",
        **kwargs
    ):
        self.smooth_fun = self._get_fun(method)
        self.method_args = kwargs
    
    def _get_fun(self, method):
        if callable(method):
            return method
        if method == "dct_smooth":
            return dct_smooth
        if method == "dct_smooth_regression":
            return dct_smooth_regression
        
    def smooth(
            self, 
            x: np.array
        ) -> Smoothed:
        """Apply the smoother function to the data

        Args:
            x (np.array): a 1D numpy array

        Returns:
            (Smoothed): A `Smoothed` object
        """
        return self.smooth_fun(x, **self.method_args)

def dct_smooth(
        x:np.array, 
        order:int = 5
    ) -> Smoothed:
    """A DCT Smoother

    Args:
        x (np.array): A 1D array of values to smooth.
        order (int, optional): DCT Order. Defaults to 5.

    Returns:
        (Smoothed): See `Smoothed`
    """
    coefs = scipy.fft.dct(x)
    coef_subset = coefs[0:order]
    smooth = scipy.fft.idct(coef_subset, n = x.shape[0])
    return Smoothed(
        smoothed=smooth, 
        params=coef_subset
    )


def dct_smooth_regression(
        x:np.array, 
        order:int = 5
    ) -> Smoothed:
    """A DCT Smoother using regression

    Args:
        x (np.array): A 1D array to smooth
        order (int, optional): Order of the DCT smoother. 
             Defaults to 5.

    Returns:
        (Smoothed): See `smoothed`
    """

    y = np.array (x)
    N = x.size
    predictors = np.array (
        [(np.cos(np.pi * (np.arange(N)/N) * k)) 
         for k in range(order)]
        )

    nan_entries = np.isnan(y)
    
    y_to_fit = y[~nan_entries]
    predictors_to_use = predictors[:,~nan_entries].T

    try:
        coefs = np.dot(
            (np.linalg.inv(
                np.dot(predictors_to_use.T,
                    predictors_to_use)
                )
            ), 
            np.dot(
                predictors_to_use.T,
                y_to_fit)
            )
        smooth = np.dot(predictors.T, coefs)
    except:
        smooth = np.full(y.shape, np.nan)
        coefs = np.full((order,), np.nan)
    return Smoothed(
        smoothed=smooth,
        params = coefs
    )