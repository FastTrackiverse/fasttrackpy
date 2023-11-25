from fasttrackpy.cli import fasttrack
from pathlib import Path
from click.testing import CliRunner
import pytest


class TestCLI:
    sound_path = Path("tests", "test_data", "ay.wav")
    audio_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.wav")
    tg_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid")
    def test_file_usage(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()

        runner = CliRunner()
        runner.invoke(
            fasttrack,
            f"audio --file {str(self.sound_path)} --dest {str(out_dir)}"
        )
        out_files = list(out_dir.glob("*"))
        assert all([x.is_file() for x in out_files])
        [x.unlink() for x in out_files]
        out_dir.rmdir()


    def test_dir_usage(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()

        runner = CliRunner()
        runner.invoke(
            fasttrack,
            f"audio --dir {str(self.sound_path.parent)} --dest {str(out_dir)}"
        )
        
        out_files = list(out_dir.glob("*"))
        assert all([x.is_file() for x in out_files])

        [x.unlink() for x in out_files]
        out_dir.rmdir()

    def test_audio_tg(self):
        out_file = Path("tests", "test_data", "vowels.csv")
        runner = CliRunner()
        runner.invoke(
            fasttrack,
            f"audio-textgrid --audio {str(self.audio_path)} --textgrid {str(self.tg_path)} --target-tier Phone --target-labels AY --output {str(out_file)}"
        )

        assert out_file.exists()

        out_file.unlink()

