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

SOUND_PATH = Path("tests", "test_data", "ay.wav")
SOUND_DIR = Path("tests", "test_data")
SOUND = pm.Sound(str(SOUND_PATH))


class TestProcessAudio:

    def test_process_audio(self):
        candidates = process_audio_file(SOUND_PATH)
        assert isinstance(candidates, CandidateTracks)
        assert candidates.file_name == str(SOUND_PATH.name)

class TestProcessDirectory:

    def test_process_directory(self):
        candidate_list = process_directory(SOUND_DIR)
        assert isinstance(candidate_list, list)
        assert all(
            [isinstance(x, CandidateTracks)
             for x in candidate_list]
        )

        assert candidate_list[0].file_name != \
                candidate_list[1].file_name