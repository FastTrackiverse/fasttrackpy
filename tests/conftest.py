from pathlib import Path
import pytest
import parselmouth as pm
from fasttrackpy import CandidateTracks, process_audio_textgrid

TEST_DATA = pytest.mark.datafiles(
  Path("tests", "test_data", "ay.wav"),
  Path("tests", "test_data", "aw.wav"),
  Path("tests", "test_data", "ə.wav"),
  Path("tests", "test_data", "config.yml")  
)


CORPUS = pytest.mark.datafiles(
  Path("tests", "test_data", "corpus")
)

@pytest.fixture
@TEST_DATA
def sound(datafiles):
    sound = pm.Sound(str(datafiles/"ay.wav"))
    return sound

@pytest.fixture
def candidates(sound):
    candidates = CandidateTracks(
            sound = sound
        )
    return candidates

@pytest.fixture
@CORPUS
def candidates2(datafiles):
  cands = process_audio_textgrid(
     audio_path=datafiles/"josef-fruehwald_speaker.wav",
     textgrid_path=datafiles/"josef-fruehwald_speaker.TextGrid"
  )
  return cands
