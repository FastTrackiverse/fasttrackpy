from fasttrackpy.processors.smoothers import Smoothed,\
                                             Smoother
from scipy.fft import dct, idct
import numpy as np

class TestSmoother:
    coefs = np.array([5,10,5,10,5])
    order = 5
    n = 100
    x = idct(coefs, n=n)

    def test_smoother_default(self):
        this_smoother = Smoother(order = self.order)
        smoothed = this_smoother.smooth(self.x)

        assert isinstance(smoothed, Smoothed)
        assert smoothed.smoothed.shape == (self.n,)
        assert smoothed.params.shape == self.coefs.shape
        assert np.all(
            np.isclose(smoothed.params, self.coefs)
        )

    def test_smoother_regression(self):
        this_smoother = Smoother(
            method = "dct_smooth_regression",
            order = self.order
        )
        smoothed = this_smoother.smooth(self.x)

        assert isinstance(smoothed, Smoothed)
        assert smoothed.smoothed.shape == (self.n,)
        assert smoothed.params.shape == self.coefs.shape

    def test_custom_regression(self):
        def mean_smooth(x):
            mean_value = np.mean(x)
            out = np.ones(x.shape) * mean_value
            return Smoothed(
                smoothed = out,
                params = mean_value
            )
        
        this_smoother = Smoother(method = mean_smooth)
        smoothed = this_smoother.smooth(self.x)

        assert isinstance(smoothed, Smoothed)
        assert smoothed.smoothed.shape == (self.n,)