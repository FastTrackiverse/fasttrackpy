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

    def test_nformant(self):
        track_high = MockTrack(
            formants = np.array([1300, 2500, 3000])
        )

        f1_max = MinMaxHeuristic(
            edge="max",
            measure = "frequency",
            number = 1,
            boundary=1200
        )

        f4_min = MinMaxHeuristic(
            edge="min",
            measure = "frequency",
            number = 4,
            boundary=2900
        )

        f1_check = f1_max.eval(track_high)
        f4_check = f4_min.eval(track_high)

        assert not np.isfinite(f1_check)
        assert f4_check == 0



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


class TestSpacing:

    def test_f3_spacing(self):
        low_f3_close = MockTrack(
            formants = np.array([100, 200, 1000])
        )

        low_f3_far = MockTrack(
            formants = np.array([100, 700, 1000])
        )

        high_f3_close = MockTrack(
            formants = np.array([100, 200, 3000])
        )

        high_f3_far = MockTrack(
            formants = np.array([100, 700, 3000])
        )

        f3_spacing = SpacingHeuristic(
            top = [3],
            bottom = [1,2],
            top_diff=2000,
            bottom_diff=500
        )

        lf3c_check = f3_spacing.eval(low_f3_close)
        lf3f_check = f3_spacing.eval(low_f3_far)
        hf3c_check = f3_spacing.eval(high_f3_close)
        hf3f_check = f3_spacing.eval(high_f3_far)

        assert not np.isfinite(lf3c_check)
        assert lf3f_check == 0
        assert hf3c_check == 0
        assert hf3f_check == 0

    def test_f3f4_spacing(self):
        close_close = MockTrack(
            formants = np.array([100, 200, 2000, 2400])
        )
        close_far = MockTrack(
            formants = np.array([100, 1700, 2000, 2400])
        )
        
        far_close = MockTrack(
            formants = np.array([100, 200, 2000, 3000])
        )

        far_far = MockTrack(
            formants = np.array([100, 1700, 2000, 3000])
        )

        f3f4_space = SpacingHeuristic(
            top = [3,4],
            bottom=[1,2],
            top_diff=500,
            bottom_diff=1500
        )

        cc_check = f3f4_space.eval(close_close)
        cf_check = f3f4_space.eval(close_far)
        fc_check = f3f4_space.eval(far_close)
        ff_check = f3f4_space.eval(far_far)

        assert not np.isfinite(cc_check)
        assert cf_check == 0
        assert fc_check == 0
        assert ff_check == 0