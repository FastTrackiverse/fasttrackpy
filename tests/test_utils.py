from fasttrackpy.utils.safely import safely, filter_nones

class TestSafely:

    @safely()
    def add_2(self, x):
        return x+2
    
    def test_safely(self):
        a = [1, 2, "c", 6]
        result = [self.add_2(x) for x in a]

        assert result[2] is None

    def test_filter_nones(self):
        a = [1, 2, "c", 6]
        b = ["a", "b", "c", "d"]
        result = [self.add_2(x) for x in a]

        result, b = filter_nones(result, [result, b])

        assert len(result) == len(b) == 3
        assert not "c" in b