from fasttrackpy.tracks import Track,\
                               OneTrack,\
                               CandidateTracks,\
                               Smoother,\
                               Loss, \
                               Agg
from fasttrackpy.patterns.corpus import process_corpus,\
    get_audio_files, \
    get_corpus, \
    read_and_associate_tg, \
    get_target_tiers, \
    get_target_intervals
from fasttrackpy.processors.outputs import write_data
from aligned_textgrid import SequenceInterval, \
    AlignedTextGrid, \
    SequenceTier

import parselmouth as pm
import polars as pl
import numpy as np
from pathlib import Path


class TestHelpers:

    def test_get_audio(self):
        all_audio = get_audio_files(Path("tests", "test_data", "corpus"))
        assert len(all_audio) == 2

    def test_get_corpus(self):
        all_audio = get_audio_files(Path("tests", "test_data", "corpus"))
        corpus = get_corpus(all_audio)

        assert len(corpus) == 2
        assert all([x.wav.exists for x in corpus])
        assert all([x.tg.exists for x in corpus])

    def test_get_tgs(self):

        all_audio = get_audio_files(
            Path("tests", "test_data", "corpus")
            )
        corpus = get_corpus(all_audio)
        all_tg = [
            read_and_associate_tg(pair) 
            for pair in corpus
            ]     

        assert all([isinstance(x, AlignedTextGrid)
                    for x in all_tg])
        
        assert all([hasattr(x, "wav") for x in all_tg])

    def test_get_target_tiers(self):
        all_audio = get_audio_files(
            Path("tests", "test_data", "corpus")
            )
        corpus = get_corpus(all_audio)
        all_tg1 = [
            read_and_associate_tg(pair) 
            for pair in corpus
            ]
        
        all_tiers1 = [
            get_target_tiers(tg) 
            for tg in all_tg1
        ]

        assert isinstance(all_tiers1[0][0], SequenceTier)

        all_tg2 = [
            read_and_associate_tg(
                pair, 
                entry_classes=[SequenceInterval]
                ) 
            for pair in corpus
            ]
        
        all_tiers2 = get_target_tiers(
            all_tg2[1], 
            target_tier="phones"
            )

        assert isinstance(all_tiers2[0], SequenceTier)

    def test_get_target_intervals(self):
        all_audio = get_audio_files(
            Path("tests", "test_data", "corpus")
            )
        
        corpus = get_corpus(all_audio)

        all_tg = [
            read_and_associate_tg(pair) 
            for pair in corpus
            ]
        
        all_tiers = [
            get_target_tiers(tg) 
            for tg in all_tg
        ]

        all_intervals = [
            get_target_intervals(tiers)
                for tiers in all_tiers
            ]
        
        assert [isinstance(interval, SequenceInterval)
                for group in all_intervals 
                for interval in group
                ]
        
        assert [hasattr(interval, "wav")
                for group in all_intervals 
                for interval in group
                ]

class TestCorpus:

    def test_corpus_process(self):

        all_candidates = process_corpus(Path("tests", "test_data", "corpus"))

        assert [isinstance(cand, CandidateTracks) for cand in all_candidates]