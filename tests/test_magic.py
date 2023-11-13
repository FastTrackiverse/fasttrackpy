try:
    import magic
except:
    no_magic = True
    import sndhdr
from pathlib import Path
import pytest

class TestMagicImport:
    @pytest.mark.skipif(no_magic, reason = "libmagic unavailable")
    def test_magic_import(self):
        m = magic.Magic()
        assert m
        audio_path = Path("tests", "test_data", "ay.wav")
        x = magic.from_file(audio_path, mime=True)
        assert x
        assert "audio" in x

