from pathlib import Path
import pytest
from fasttrackpy.patterns.just_audio import is_audio

from .conftest import TEST_DATA

@TEST_DATA
class TestAudioCheck():
    def test_audio_files(self, datafiles):
        wavs = datafiles.glob("*.wav")
        for w in wavs:
            assert is_audio(w)
    
    def test_all_files(self, datafiles):
        all_files = datafiles.glob("*")
      

        all_files = [f for f in all_files if f.is_file()]
        assert len(all_files) > 0

        for f in all_files:
            if f.suffix == ".wav":
                assert is_audio(f)
            else:
                assert not is_audio(f)