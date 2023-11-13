try:
    import magic
    no_magic = False
except:
    no_magic = True
    import sndhdr
from pathlib import Path
import pytest
from fasttrackpy.patterns.just_audio import create_audio_checker,\
                                            is_audio

class TestMagicImport:
    @pytest.mark.skipif(no_magic, reason = "libmagic unavailable")
    def test_magic_import(self):
        audio_path = Path("tests", "test_data", "ay.wav")
        x = magic.from_file(str(audio_path), mime=True)
        assert x
        assert "audio" in x

    def test_conditional_checker(self):
        checker = create_audio_checker(no_magic=no_magic)
        assert checker
        audio_path = Path("tests", "test_data", "ay.wav")
        is_audio = checker(str(audio_path))
        assert is_audio

    def test_universal_checker(self):
        audio_path = Path("tests", "test_data", "ay.wav")
        assert is_audio(str(audio_path))
