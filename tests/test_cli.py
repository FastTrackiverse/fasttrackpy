from fasttrackpy.cli import fasttrack
from pathlib import Path
from click.testing import CliRunner
import pytest
import yaml
import logging

from .conftest import TEST_DATA, CORPUS
@pytest.fixture(scope="session")
def dest(tmp_path_factory):
    dest = tmp_path_factory.mktemp("dest") 
    return dest


class TestCLI:
    # sound_path = Path("tests", "test_data", "ay.wav")
    # audio_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.wav")
    # tg_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid")
    # corpus_path = Path("tests", "test_data", "corpus")
    
    @TEST_DATA
    def test_file_usage(self, datafiles, dest):
        out_dir = dest

        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--file", datafiles/"ay.wav", 
             "--dest", out_dir]
        )

        assert result.exit_code == 0, result.output


    @TEST_DATA
    def test_file_usage_heuristic(self, datafiles, dest):
        out_dir = dest
        print(datafiles.joinpath("ay.wav").exists())
        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--file", str(datafiles/"ay.wav"), 
             "--dest", str(out_dir), 
             "--f1-max-heuristic"]
        )
    
        assert result.exit_code == 0, result.output


    @TEST_DATA
    def test_config_file(self, datafiles):
        config_path = datafiles/"config.yml"
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


    @TEST_DATA
    def test_dir_usage(self, datafiles, dest):
        out_dir = dest

        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio", 
             "--dir", str(datafiles),
               "--dest", str(out_dir)]
        )
        
        assert result.exit_code == 0, result.output


    @CORPUS
    def test_audio_tg(self, datafiles, dest):
        out_dir = dest
        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["audio-textgrid", 
             "--audio", datafiles/"josef-fruehwald_speaker.wav", 
             "--textgrid", datafiles/"josef-fruehwald_speaker.TextGrid", 
             "--target-tier", "Phone", 
             "--target-labels", "AY",
             "--dest", out_dir]
        )

        assert result.exit_code == 0, result.output

    @CORPUS
    def test_corpus(self, datafiles, dest):
        out_dir = dest
        runner = CliRunner()
        result = runner.invoke(
            fasttrack,
            ["corpus",
             "--corpus", datafiles,
             "--target-labels", "AY",
             "--dest", out_dir, 
             "--separate-output"]
        )

        assert result.exit_code == 0, result.output

