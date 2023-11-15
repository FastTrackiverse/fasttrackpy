import numpy as np
from fasttrackpy.processors.aggs import Agg, agg_sum

class TestAggs:

    def agg_prod(x, axis = 0):
        result = np.prod(x, axis=axis)
        return result

    def test_agg_sum(self):
        x = np.array([1, 2, 3])
        out = agg_sum(x)

        assert np.isclose(out, 6)
    
    def test_Agg_class_default(self):
        this_agg = Agg(method = "agg_sum")
        x = np.array([1, 2, 3])
        out = this_agg.aggregate(x)

        assert np.isclose(out, 6)

    def test_custom_agg(self):

        def agg_prod(x, axis = 0):
            result = np.prod(x, axis=axis)
            return result
        
        this_agg = Agg(method = agg_prod)
        x = np.array([2,2,2])
        out = this_agg.aggregate(x)
        assert np.isclose(out, 8)