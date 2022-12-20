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