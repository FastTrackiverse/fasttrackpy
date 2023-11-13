import magic
from pathlib import Path

class TestMagicImport:

    def test_magic_import(self):
        m = magic.Magic()
        assert m
        audio_path = Path("tests", "test_data", "ay.wav")
        x = magic.from_file(audio_path, mime=True)
        assert x
        assert "audio" in x

