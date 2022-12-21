import numpy as np
import scipy.fft

def dct_smooth(x, order = 5, out = "smooth"):
    """
    DCT smoother
    """
    coefs = scipy.fft.dct(x)
    coef_subset = coefs[0:order]
    smooth = scipy.fft.idct(coef_subset, n = x.shape[0])
    if out == "smooth":
        return(smooth)
    elif out == "coef":
        return(coef_subset)
    elif out == "both":
        return(coef_subset, smooth)



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
        return(smooth)
    elif out == "coef":
        return(coef_subset)
    elif out == "both":
        return(coef_subset, smooth)