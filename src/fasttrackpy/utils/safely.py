import warnings
from typing import Callable
from typing import Sequence


def safely(
        message: str = f"There was a problem a function's application."
    ):
    """
    A decorator for more graceful failing. 
    If the decorated function raises an exception, 
    it will return `None`. 

    Args:
        message (str, optional): 
            A warning message in the case of an exception. 
            Defaults to `f"There was a problem a function's application."`.
    """
    def decorator(func:Callable):
        def safe_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                warnings.warn(message)
                return None
        return safe_func
    return decorator

def filter_nones(
    filterer: Sequence,
    to_filter: list[Sequence]
):
    nones = [i for i,x in enumerate(filterer) if x is None]
    filtered = [
        [x for i,x in enumerate(one_list) if not i in nones]
        for one_list in to_filter
    ]
    return filtered

