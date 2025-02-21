from fasttrackpy.processors.heuristic import MinMaxHeuristic, SpacingHeuristic
import numpy as np

class MockTrack:
    def __init__(
            self,
            formants = np.array([]),
            bandwidths=np.array([])
        ):
        self.n_formants = np.array([
            formants.size,
            bandwidths.size
        ]).max()
        self.log_parameters = np.log(
            formants.reshape(-1, 1)
        )/np.sqrt(2)
        self.bandwidth_parameters = np.log(
            bandwidths.reshape(-1, 1)
        )/np.sqrt(2)

class TestMinMax:

    def test_f1_max(self):
        track_high = MockTrack(
            formants = np.array([1300, 2500, 3000])
        )

        track_low = MockTrack(
            formants = np.array([1100, 2500, 3000])
        )

        f1_max = MinMaxHeuristic(
            edge="max",
            measure = "frequency",
            number = 1,
            boundary=1200
        )

        high_check = f1_max.eval(track_high)
        low_check = f1_max.eval(track_low)

        assert not np.isfinite(high_check)
        assert low_check == 0

    def test_f2_bandwidth(self):
        f2_band_high = MockTrack(
            bandwidths=np.array([200, 550, 400])
        )

        f2_band_low = MockTrack(
            bandwidths=np.array([200, 300, 400])
        )

        b2_max = MinMaxHeuristic(
            edge = "max",
            measure="bandwidth",
            number = 2,
            boundary=500
        )

        high_check = b2_max.eval(f2_band_high)
        low_check = b2_max.eval(f2_band_low)

        assert not np.isfinite(high_check)
        assert low_check == 0

    def test_f3_bandwidth(self):
        f3_band_high = MockTrack(
            bandwidths=np.array([200, 300, 620])
        )

        f3_band_low = MockTrack(
            bandwidths=np.array([200, 300, 400])
        )

        b3_max = MinMaxHeuristic(
            edge = "max",
            measure="bandwidth",
            number=3,
            boundary=600
        )

        high_check = b3_max.eval(f3_band_high)
        low_check = b3_max.eval(f3_band_low)

        assert not np.isfinite(high_check)
        assert low_check == 0

    def test_f4_min(self):
        f4_high = MockTrack(
            formants=np.array([100, 200, 300, 3000])
        )
        f4_low = MockTrack(
            formants=np.array([100, 200, 300, 2000])
        )

        f4_min = MinMaxHeuristic(
            edge="min",
            measure="frequency",
            number=4,
            boundary=2900
        )

        high_check = f4_min.eval(f4_high)
        low_check = f4_min.eval(f4_low)

        assert high_check == 0
        assert not np.isfinite(low_check)




