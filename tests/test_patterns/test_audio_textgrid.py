from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg
from fasttrackpy.processors.heuristic import (
    MinMaxHeuristic,
    SpacingHeuristic,
    F1_Max, 
    B2_Max,
    B3_Max,
    F4_Min,
    Rhotic,
    F3_F4_Sep
)
from fasttrackpy.patterns.audio_textgrid import process_audio_textgrid
from fasttrackpy.processors.outputs import write_data
from aligned_textgrid import SequenceInterval

import parselmouth as pm
import polars as pl
import numpy as np
from pathlib import Path

TG_PATH = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid")
AUDIO_PATH = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.wav")

class TestAudioTG:

    def test_audio_tg(self):
        candidates = process_audio_textgrid(
            audio_path=AUDIO_PATH,
            textgrid_path=TG_PATH
        )
        assert isinstance(candidates[0], CandidateTracks)
        assert candidates[0].winner.group
        assert isinstance(candidates[0].interval, SequenceInterval)

        df = candidates[0].to_df()
        assert "group" in df.columns
        assert "id" in df.columns

    def test_audio_tg_heuristics(self):
        bad_heuristic = MinMaxHeuristic(
            edge="max",
            measure="frequency",
            number=1,
            boundary=600
        )
        candidates = process_audio_textgrid(
            audio_path=AUDIO_PATH,
            textgrid_path=TG_PATH,
            heuristics=[bad_heuristic]
        )
        assert isinstance(candidates[0], CandidateTracks)
        assert isinstance(candidates[0].heuristics, list)
        assert isinstance(candidates[0].heuristics[0], (MinMaxHeuristic|SpacingHeuristic))

        all_error = np.array([
            cand2.total_error.max()
            for cand in candidates
            for cand2 in cand.candidates
        ])

        assert np.any(~np.isfinite(all_error))

    def test_audio_tg2(self)    :
        candidates = process_audio_textgrid(
            audio_path=AUDIO_PATH,
            textgrid_path=TG_PATH,
            entry_classes="SequenceInterval",
            target_tier="phones",
            target_labels="AY"
        )

        assert isinstance(candidates[0], CandidateTracks)
        assert candidates[0].winner.group
        assert all(["AY" in x.winner.label for x in candidates])
        out_file = Path("tests", "test_data", "output.csv")
        write_data(candidates, file = out_file)

        assert out_file.exists()

        out_file.unlink()