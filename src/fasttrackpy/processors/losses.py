import numpy as np
import numpy.typing as npt
from collections.abc import Callable

class Loss:
    """_summary_

    Args:
        method (Union[str, Callable], optional): _description_. Defaults to "lmse".
    """

    def __init__(
        self,
        method: str|Callable = "lmse",
        **kwargs
    ):
        self.method = self._get_fun(method)
        self.method_args = kwargs
    
    def _get_fun(
            self, 
            method: str|Callable 
        ) -> Callable:
        if callable(method):
            return method
        elif method == "lmse":
            return lmse
        else:
            return mse
        
    def calculate_loss(
        self,
        formants: np.ndarray, 
        smoothed: np.ndarray
    ):
        return self.method(formants, smoothed, **self.method_args)
    
def lmse(
        formants: npt.NDArray, 
        smoothed: npt.NDArray, 
        axis: int = 1
    ) -> npt.NDArray:
    """_summary_

    Args:
        formants (npt.NDArray): _description_
        smoothed (npt.NDArray): _description_
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        npt.NDArray: _description_
    """
    sqe = np.power(np.log(formants) - np.log(smoothed), 2)
    mse = np.nanmean(sqe, axis = axis)
    return mse

def mse(
        formants: npt.NDArray, 
        smoothed: npt.NDArray, 
        axis: int = 1
    ) -> npt.NDArray:
    """_summary_

    Args:
        formants (npt.NDArray): _description_
        smoothed (npt.NDArray): _description_
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        npt.NDArray: _description_
    """
    sqe = np.power(formants - smoothed, 2)
    mse = np.nanmean(sqe, axis = axis)
    return mse