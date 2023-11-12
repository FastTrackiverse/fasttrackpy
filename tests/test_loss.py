import numpy as np
from fasttrackpy.processors.losses import Loss,\
                                          mse,\
                                          lmse

class TestLoss:
    a = 10
    b = 20
    n = 10
    formants = np.ones((2,n))
    formants[0] = formants[0] * a
    formants[1] = formants[1] * b
    smoothed = np.array(formants, copy=True)
    formants[0,1] = b

    def test_loss_default(self):
        this_loss = Loss()
        expected = np.power(
            np.log(self.b)-np.log(self.a),
            2
        )/self.n

        loss = this_loss.calculate_loss(self.formants, self.smoothed)

        assert np.isclose(loss[0], expected)

    def test_loss_mse(self):
        this_loss = Loss(method = 'mse')
        expected = np.power(self.b-self.a, 2)/self.n

        loss = this_loss.calculate_loss(self.formants, self.smoothed)

        assert np.isclose(loss[0], expected)
    
    def test_custom_error(self):
        def mae(
                formants,
                smoothed,
                axis = 1
        ):
            abs_e = np.abs(formants-smoothed)
            mae = np.nanmean(abs_e, axis = axis)
            return mae
        
        this_loss = Loss(method = mae)
        expected = np.abs(self.b - self.a)/self.n

        loss = this_loss.calculate_loss(self.formants, self.smoothed)
        
        assert np.isclose(loss[0], expected)