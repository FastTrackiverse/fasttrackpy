import numpy as np

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