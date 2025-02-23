from fasttrackpy.cli import fasttrack
from pathlib import Path
from click.testing import CliRunner
import pytest
import yaml
import logging

class TestCLI:
    sound_path = Path("tests", "test_data", "ay.wav")
    audio_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.wav")
    tg_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid")
    corpus_path = Path("tests", "test_data", "corpus")

    def test_file_usage(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--file", self.sound_path, 
             "--dest", out_dir]
        )

        assert result.exit_code == 0, result.output
        out_files = list(out_dir.glob("*"))
        [x.unlink() for x in out_files]
        out_dir.rmdir()

    def test_file_usage_heuristic(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--file", self.sound_path, 
             "--dest", out_dir, 
             "--f1-max-heuristic"]
        )

        assert result.exit_code == 0, result.output
        out_files = list(out_dir.glob("*"))
        [x.unlink() for x in out_files]
        out_dir.rmdir()        

    def test_config_file(self):
        config_path = Path("tests", "test_data", "config.yml")
        with config_path.open() as file:
            params = yaml.safe_load(file)

        sound_path = Path(params["file"])
        dest = Path(params["dest"])
        if not dest.is_dir():
            dest.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--config", config_path]
        )

        assert result.exit_code == 0, result.output
        out_files = list(dest.glob("*"))
        [x.unlink() for x in out_files]
        dest.rmdir()


    def test_dir_usage(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--dir", self.sound_path.parent,
               "--dest", out_dir]
        )
        
        assert result.exit_code == 0, result.output
        out_files = out_dir.glob("*")
        [x.unlink() for x in out_files]
        out_dir.rmdir()

    def test_audio_tg(self):
        out_dir = self.sound_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio-textgrid", 
             "--audio", self.audio_path, 
             "--textgrid", self.tg_path, 
             "--target-tier", "Phone", 
             "--target-labels", "AY",
             "--dest", out_dir]
        )

        assert result.exit_code == 0, result.output
        out_files = list(out_dir.glob("*"))

        [x.unlink() for x in out_files]
        out_dir.rmdir()

    def test_corpus(self):
        out_dir = self.corpus_path.parent.joinpath("output")
        if not out_dir.is_dir():
            out_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["corpus",
             "--corpus", self.corpus_path,
             "--target-labels", "AY",
             "--dest", out_dir, 
             "--separate-output"]
        )

        assert result.exit_code == 0, result.output
        out_files = list(out_dir.iterdir())
        assert len(out_files) > 1

        [x.unlink() for x in out_files]
        out_dir.rmdir()
