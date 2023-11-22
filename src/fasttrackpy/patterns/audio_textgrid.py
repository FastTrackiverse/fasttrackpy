from aligned_textgrid import AlignedTextGrid, Word, Phone, SequenceInterval
import aligned_textgrid
from fasttrackpy import CandidateTracks, Smoother, Loss, Agg
from fasttrackpy.patterns.just_audio import create_audio_checker

from pathlib import Path
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
        window_length: float = 0.05,
        time_step: float = 0.002,
        pre_emphasis_from: float = 50,
        smoother: Smoother = Smoother(),
        loss_fun: Loss = Loss(),
        agg_fun: Agg = Agg()
)->CandidateTracks:
    
    if not is_audio(str(audio_path)):
        raise TypeError(f"The file at {str(audio_path)} is not an audio file")
    
    
    entry_classes = get_interval_classes(textgrid_format=textgrid_format)
    tg = AlignedTextGrid(textgrid_path=textgrid_path, entry_classes=entry_classes)
    return tg
