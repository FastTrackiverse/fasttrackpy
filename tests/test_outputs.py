import pytest
from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg
from fasttrackpy.processors.outputs import write_data, \
    pickle_candidates,\
    unpickle_candidates
from fasttrackpy.patterns.just_audio import process_audio_file
from fasttrackpy import process_audio_textgrid
import parselmouth as pm
import polars as pl
import numpy as np
import matplotlib.pyplot as mp
from pathlib import Path
from PIL import Image

SOUND_PATH = Path("tests", "test_data", "ay.wav")
SOUND = pm.Sound(str(SOUND_PATH))

class TestDataFrames:
    candidates = CandidateTracks(sound = SOUND)
    def test_formant_df(self):
       
        formant_df = self.candidates.to_df(output="formants")
        big_formant_df = self.candidates.to_df(which='all', output="formants")

        assert isinstance(formant_df, pl.DataFrame)
        assert isinstance(big_formant_df, pl.DataFrame)

    def test_param_df(self):
        param_df = self.candidates.to_df(output="param")
        big_param_df = self.candidates.to_df(which='all', output="formants")

        assert isinstance(param_df, pl.DataFrame)
        assert isinstance(big_param_df, pl.DataFrame)

    def test_log_param_df(self):
        log_param_df = self.candidates.to_df(output="log_param")
        big_log_param_df = self.candidates.to_df(which='all', output='log_param')

        assert isinstance(log_param_df, pl.DataFrame)                
        assert isinstance(big_log_param_df, pl.DataFrame)


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

    def test_save_spectrogram(self, tmp_path):
        plot_file = tmp_path / "test.png"
        width = 8
        height = 5
        dpi = 100
        self.cands.winner.spectrogram(
            figsize = (width, height),
            file_name = plot_file,
            dpi = dpi
        )
        assert plot_file.exists()
        image = Image.open(plot_file)
        im_width, im_height = image.size
        assert im_width == width * dpi
        assert im_height == height * dpi

    def test_save_cand_spectrogram(self, tmp_path):
        plot_file = tmp_path / "test2.png"
        width = 8
        height = 5
        dpi = 100
        self.cands.spectrograms(
            figsize = (width, height),
            file_name = plot_file,
            dpi = dpi
        )
        assert plot_file.exists()
        image = Image.open(plot_file)
        im_width, im_height = image.size
        assert im_width <= width * dpi
        assert im_height <= height * dpi

class TestPickle:
    cands = CandidateTracks(SOUND)
    

    def test_pickle_unpickle(self, tmp_path):
        pickle_file = tmp_path / "cand.pkl"
        pickle_candidates(self.cands, file = pickle_file)
        assert pickle_file.exists()

        re_read = unpickle_candidates(file = pickle_file)
        assert isinstance(re_read, CandidateTracks)
        assert np.isclose(
            re_read.winner.maximum_formant,
            self.cands.winner.maximum_formant
            )
        assert len(re_read.candidates) == len(self.cands.candidates)

    def test_big_pickle_unpickle(self, tmp_path):
        cands2 = process_audio_textgrid(
            audio_path=Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.wav"),
            textgrid_path=Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid"),
        )
        pickle_file = tmp_path / "cand2.pkl"
        pickle_candidates(cands2[0], file = pickle_file)
        assert pickle_file.exists()

        re_read = unpickle_candidates(file = pickle_file)
        assert len(re_read.candidates) == len(cands2[0].candidates)