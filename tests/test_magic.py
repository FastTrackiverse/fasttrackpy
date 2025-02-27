# try:
#     import magic
#     no_magic = False
# except:
#     no_magic = True
#     import sndhdr
from pathlib import Path
import pytest
from fasttrackpy.patterns.just_audio import is_audio

class TestAudioCheck():
    def test_audio_files(self):
        data_path = Path("tests", "test_data")
        wavs = data_path.glob("*.wav")
        for w in wavs:
            assert is_audio(w)
    
    def test_all_files(self):
        data_path = Path("tests", "test_data")
        all_files = data_path.glob("*")
      

        all_files = [f for f in all_files if f.is_file()]
        assert len(all_files) > 0

        for f in all_files:
            if f.suffix == ".wav":
                assert is_audio(f)
            else:
                assert not is_audio(f)