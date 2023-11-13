from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg

import parselmouth as pm
import polars as pl
import numpy as np
from pathlib import Path

SOUND_PATH = Path("tests", "test_data", "ay.wav")
SOUND = pm.Sound(str(SOUND_PATH))

class TestTrack:

    def test_track_default(self):
        this_track = Track(sound=SOUND)
        assert this_track
        assert isinstance(this_track.sound, pm.Sound)
        assert this_track.n_formants == 4
        assert np.isclose(this_track.window_length, 0.05)
        assert np.isclose(this_track.time_step, 0.002)
        assert np.isclose(this_track.pre_emphasis_from, 50)
        assert isinstance(this_track.smoother, Smoother)
        assert isinstance(this_track.loss_fun, Loss)
        assert isinstance(this_track.agg_fun, Agg)


class TestOneTrack:

    def test_one_track_default(self):
        this_track = OneTrack(
            sound = SOUND,
            maximum_formant=4000
        )

        assert this_track
        assert this_track.formants.shape[0] == 4