import parselmouth as pm
from aligned_textgrid import AlignedTextGrid, Word, Phone, SequenceInterval, SequenceTier
import aligned_textgrid
from fasttrackpy import CandidateTracks, Smoother, Loss, Agg
from fasttrackpy.patterns.just_audio import create_audio_checker
import re

from pathlib import Path
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

try:
    import magic
    no_magic = False
except:
    warnings.warn("libmagic not found. "\
                "Some audio file types won't be discovered by fasttrack. "\
                "(mp3, ogg, ...)")
    import sndhdr
    from sndhdr import SndHeaders
    no_magic = True

is_audio = create_audio_checker(no_magic=no_magic)

def get_interval_classes(
        textgrid_format: list = ["Word", "Phone"]
        ):
    if all(
        [hasattr(aligned_textgrid, x) for x in textgrid_format]
    ):
        return [getattr(aligned_textgrid, x) for x in textgrid_format]
    
    return SequenceInterval

def get_target_tiers(
        tg: AlignedTextGrid,
        target_tier: str = "Phone"
    ):

    if all(
        [hasattr(group, target_tier) for group in tg]
    ):
        return [getattr(group, target_tier) for group in tg]
    
    tier_names = [tier.name for group in tg for tier in group]
    if target_tier in tier_names:
        return [tier for group in tg for tier in group if tier.name == target_tier]
    
    raise Exception(f"Could not {target_tier} target tier in textgrid")

def get_target_intervals(
        target_tiers: list[SequenceTier],
        target_labels: "[AEIOU]"
        ):
    intervals = [
        interval 
        for tier in target_tiers 
        for interval in tier
        if re.match(target_labels, interval.label) and
        (interval.end - interval.start) > 0.05
    ]

    return intervals

def get_candidates(args_dict):
    return CandidateTracks(**args_dict)

def process_audio_textgrid(
        audio_path: str|Path,
        textgrid_path: str|Path,
        textgrid_format: list = ["Word", "Phone"],
        target_tier: str = "Phone",
        target_labels: str = "[AEIOU]",
        min_max_formant:float = 4000,
        max_max_formant:float = 7000,
        nstep:int = 20,
        n_formants: int = 4,
        window_length: float = 0.025,
        time_step: float = 0.002,
        pre_emphasis_from: float = 50,
        smoother: Smoother = Smoother(),
        loss_fun: Loss = Loss(),
        agg_fun: Agg = Agg()
)->list[CandidateTracks]:
    
    if not is_audio(str(audio_path)):
        raise TypeError(f"The file at {str(audio_path)} is not an audio file")
    
    sound = pm.Sound(str(audio_path))

    entry_classes = get_interval_classes(textgrid_format=textgrid_format)
    
    tg = AlignedTextGrid(textgrid_path=textgrid_path, entry_classes=entry_classes)
    target_tiers = get_target_tiers(tg, target_tier=target_tier)
    target_intervals = get_target_intervals(
        target_tiers=target_tiers, 
        target_labels=target_labels
        )
    sound_parts = [
        sound.extract_part(from_time = interval.start, to_time = interval.end)
        for interval in target_intervals
    ]

    arg_list = [
        {
            "sound": x,
            #"interval": interval,
            "min_max_formant": min_max_formant,
            "max_max_formant": max_max_formant,
            "nstep": nstep,
            "n_formants": n_formants,
            "window_length": window_length,
            "time_step" : time_step,
            "pre_emphasis_from": pre_emphasis_from,
            "smoother": smoother,
            "loss_fun":loss_fun,
            "agg_fun": agg_fun
        } for x, interval in zip(sound_parts, target_intervals)
    ]
    
    #n_jobs = multiprocessing.cpu_count()
    candidate_list = [''] * len(arg_list)
    for idx, arg in tqdm(enumerate(arg_list)):
        candidate_list[idx] = get_candidates(arg)
        candidate_list[idx].interval = target_intervals[idx]
    
    return candidate_list


