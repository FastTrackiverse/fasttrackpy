import numpy as np
import numpy.typing as npt
from typing import Union
from collections.abc import Callable

class Agg:
    """_summary_

    Args:
        method (Union[str, Callable], optional): _description_. Defaults to "agg_sum".
    """

    def __init__(
        self, 
        method: str|Callable = "agg_sum",
        **kwargs
    ):
        self.method = self._get_method(method)
        self.method_args = kwargs
    
    def _get_method(
        self, 
        method:str|Callable[[npt.NDArray], float]
    )->Callable:
        if callable(method):
            return method
        else:
            return agg_sum
    
    def aggregate(
        self,
        error
    ):
        return self.method(error, **self.method_args)

def agg_sum(error, axis = 0):
    """
    Sum the error
    """

    agg_error = np.sum(error, axis = axis)
    return agg_error