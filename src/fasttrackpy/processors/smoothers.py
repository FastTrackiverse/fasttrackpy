import numpy as np
import scipy.fft
from typing import Union
from collections.abc import Callable
class Smoother:
    def __init__(
        self,
        method: Union[str, Callable] = "dct_smooth",
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
        
    def smooth(self, x):
        return self.smooth_fun(x, **self.method_args)

        

def dct_smooth(x, order = 5, out = "smooth"):
    """
    DCT smoother
    """
    coefs = scipy.fft.dct(x)
    coef_subset = coefs[0:order]
    smooth = scipy.fft.idct(coef_subset, n = x.shape[0])
    if out == "smooth":
        return smooth
    if out == "coef":
        return coef_subset
    if out == "both":
        return coef_subset, smooth



def dct_smooth_regression(x, order = 5, out = "smooth"):
    """
    DCT smoother using regression
    """

    y = np.array (x)
    N = x.size
    predictors = np.array ([(np.cos(np.pi * (np.array(range(N))/N) * k)) for k in range(order)])
    predictors = predictors.T
    coefs = np.dot((np.linalg.inv(np.dot(predictors.T,predictors))), np.dot(predictors.T,y))
    smooth = np.dot(predictors, coefs)

    if out == "smooth":
        return smooth
    if out == "coef":
        return coefs
    if out == "both":
        return coefs, smooth