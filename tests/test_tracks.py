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
        td = this_track.time_domain
        assert this_track.formants.shape == (4, td.shape[0])
        assert this_track.formants.shape == this_track.smoothed_formants.shape
        assert isinstance(this_track.smooth_error, float)

        df = this_track.to_df()
        assert isinstance(df, pl.DataFrame)

    def test_custom_one_track(self):
        this_track = OneTrack(
            sound = SOUND,
            maximum_formant=5000,
            n_formants = 5,
            smoother = Smoother(order = 6)
        )

        assert this_track.formants.shape[0] == 5
        assert this_track.parameters.shape == (5, 6)

class TestCandidateTracks:

    def test_candidate_tracks_default(self):
        candidates = CandidateTracks(
            sound = SOUND
        )

        assert candidates
        assert len(candidates.candidates) == 20
        assert candidates.max_formants.shape == (20,)
        assert candidates.smooth_errors.shape == (20,)
        assert isinstance(candidates.winner, OneTrack)