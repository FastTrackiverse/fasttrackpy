from fasttrackpy.cli import fasttrack
from pathlib import Path
from click.testing import CliRunner
import pytest


class TestCLI:
    sound_path = Path("tests", "test_data", "ay.wav")
    def test_file_usage(self):
        output_file = self.sound_path.with_suffix(".csv")
        runner = CliRunner()
        runner.invoke(
            fasttrack,
            f"--file {str(self.sound_path)} --output {str(output_file)}"
        )

        assert output_file.is_file()
        output_file.unlink()


    def test_dir_usage(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()

        runner = CliRunner()
        runner.invoke(
            fasttrack,
            f"--dir {str(self.sound_path.parent)} --dest {str(out_dir)}"
        )
        
        out_files = list(out_dir.glob("*"))
        assert all([x.is_file() for x in out_files])

        [x.unlink() for x in out_files]
        out_dir.rmdir()
