import pytest
from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg
from fasttrackpy.processors.outputs import write_data
from fasttrackpy.patterns.just_audio import process_audio_file
import parselmouth as pm
import polars as pl
import numpy as np
import matplotlib.pyplot as mp
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

        assert filename1.is_file()
        assert filename2.is_file()
        assert len(alllines) > len(winnerlines)
        filename1.unlink()
        filename2.unlink()
    
    def test_write_with_dest(self):
        dest = Path("tests", "test_data")
        file_name = SOUND_PATH.with_suffix(".csv")
        candidates = CandidateTracks(sound = SOUND)
        candidates.file_name = file_name.name
        write_data(candidates=candidates,
                   destination=dest)
        
        assert file_name.is_file()
        file_name.unlink()

    def test_write_only_dest(self):
        dest =  Path("tests", "test_data")
        file_name = dest.joinpath("output.csv")
        candidates = CandidateTracks(sound = SOUND)
        write_data(candidates=candidates,
                   destination=dest)
        
        assert file_name.is_file()
        file_name.unlink()

@pytest.fixture(scope='function')
def plot_spectrogram():
    def _plot(self):
        self.cands.winner.spectrogram()
        yield mp.show()
        mp.close("all")
    return _plot

class TestPlots:

    cands = CandidateTracks(SOUND)

    def test_spectrogram(self, monkeypatch):
        monkeypatch.setattr(mp, 'show', lambda: None)
        self.cands.winner.spectrogram()
        assert True

    def test_spectrograms(self, monkeypatch):
        monkeypatch.setattr(mp, 'show', lambda: None)
        self.cands.spectrograms()
        assert True
