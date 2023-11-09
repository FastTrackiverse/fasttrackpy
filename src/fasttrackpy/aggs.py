import numpy as np

def agg_sum(error, axis = 0):
    """
    Sum the error
    """

    agg_error = np.sum(error, axis = axis)
    return agg_error