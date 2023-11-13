import warnings
from pathlib import Path
from typing import Union
from collections.abc import Callable
import parselmouth as pm
from fasttrackpy import CandidateTracks,\
                        Smoother,\
                        Loss,\
                        Agg

try:
    import magic
    no_magic = False
    warnings.warn("libmagic not found. "\
                  "Some audio file types won't be discovered by fasttrack. "\
                  "(mp3, ogg, ...)")
except:
    import sndhdr
    from sndhdr import SndHeaders
    no_magic = True

def create_audio_checker(no_magic:bool = no_magic) -> Callable:
    """Return an audio checker, dependent on 
       availability of libmagic.

    Args:
        no_magic (bool): is libmagic available

    Returns:
        (Callable): A sound file checker
    """

    def magic_checker(path: str)->bool:
        """Checks whether a file is an audio file using libmagic

        Args:
            path (str): Path to the file in question

        Returns:
            (bool): Whether or not the file is an audio file
        """
        file_mime = magic.from_file(path, mime=True)
        return "audio" in file_mime
    
    def sndhdr_checker(path: str)->bool:
        """Checks whether a file is an audio file using `sndhdr`

        Args:
            path (str): Path to the file

        Returns:
            (bool): Whether or not the file is an audio file.
        """
        hdr_info = sndhdr.what(path)
        return isinstance(hdr_info, SndHeaders)
    
    if no_magic:
        return sndhdr_checker
    
    return magic_checker

is_audio = create_audio_checker(no_magic=no_magic)

def process_audio_file(
        path: Union[str, Path],
        xmin:float = 0,
        xmax: float = None,
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
    if not is_audio(str(path)):
        raise TypeError(f"The file at {str(path)} is not an audio file")
    
    sound = pm.Sound(str(path))
    if not xmax:
        xmax = sound.xmax

    sound_to_process = sound.extract_part(from_time = xmin, to_time = xmax)
    candidates = CandidateTracks(
        sound=sound_to_process,
        min_max_formant=min_max_formant,
        max_max_formant=max_max_formant,
        nstep=nstep,
        n_formants=n_formants,
        window_length=window_length,
        time_step=time_step,
        pre_emphasis_from=pre_emphasis_from,
        smoother=smoother,
        loss_fun=loss_fun,
        agg_fun=agg_fun
    )
    candidates.file_name = Path(str(path)).name
    return candidates

def process_directory(
        path: Union[str, Path],
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
)->list[CandidateTracks]:
    if not isinstance(path, Path) and isinstance(path, str):
        path = Path(path)

    all_files = path.glob("*")
    all_audio = [x for x in all_files if is_audio(str(x))]
    all_candidates = [
        process_audio_file(
            path = x,
            min_max_formant=min_max_formant,
            max_max_formant=max_max_formant,
            nstep=nstep,
            n_formants=n_formants,
            window_length=window_length,
            time_step=time_step,
            pre_emphasis_from=pre_emphasis_from,
            smoother=smoother,
            loss_fun=loss_fun,
            agg_fun=agg_fun
        ) for x in all_audio
    ]
    for x, path in zip(all_candidates, all_audio):
        x.file_name = Path(str(path)).name

    return all_candidates

