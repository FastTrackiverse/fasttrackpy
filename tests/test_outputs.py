from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg
from fasttrackpy.processors.outputs import write_data
import parselmouth as pm
import polars as pl
import numpy as np
from pathlib import Path

SOUND_PATH = Path("tests", "test_data", "ay.wav")
SOUND = pm.Sound(str(SOUND_PATH))

class TestWrite:

    def test_formant_write(self):
        filename1 = Path("tests", "test_data", "testing.csv")
        filename2 = Path("tests", "test_data", "testing_all.csv")        
        candidates = CandidateTracks(sound = SOUND)
        write_data(candidates=candidates, file = filename1)
        write_data(candidates=candidates, file = filename2, which="all")        

        with filename1.open() as f1:
            winnerlines = f1.readlines()
        
        with filename2.open() as f2:
            alllines = f2.readlines()

        assert len(alllines) > len(winnerlines)
        filename1.unlink()
        filename2.unlink()