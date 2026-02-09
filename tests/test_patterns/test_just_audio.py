from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg
from fasttrackpy.patterns.just_audio import process_audio_file,\
                                            process_directory

import parselmouth as pm
import polars as pl
import numpy as np
from pathlib import Path

from ..conftest import TEST_DATA

SOUND_PATH = Path("tests", "test_data", "ay.wav")
SOUND_DIR = Path("tests", "test_data")
SOUND = pm.Sound(str(SOUND_PATH))


@TEST_DATA
class TestProcessAudio:

    def test_process_audio(self, datafiles):
        candidates = process_audio_file(datafiles/"ay.wav")
        assert isinstance(candidates, CandidateTracks)
        assert candidates.file_name == str(datafiles.joinpath("ay.wav").name)

@TEST_DATA
class TestProcessDirectory:

    def test_process_directory(self, datafiles):
        candidate_list = process_directory(datafiles)
        assert isinstance(candidate_list, list)
        assert all(
            [isinstance(x, CandidateTracks)
             for x in candidate_list]
        )

        assert candidate_list[0].file_name != \
                candidate_list[1].file_name