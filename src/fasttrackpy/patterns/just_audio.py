import warnings
from pathlib import Path
from typing import Union
from collections.abc import Callable
import parselmouth as pm
from fasttrackpy import CandidateTracks,\
                        Smoother,\
                        Loss,\
                        Agg

from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed

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
        file_mime = magic.from_file(str(path), mime=True)
        return "audio" in file_mime
    
    def sndhdr_checker(path: str)->bool:
        """Checks whether a file is an audio file using `sndhdr`

        Args:
            path (str): Path to the file

        Returns:
            (bool): Whether or not the file is an audio file.
        """
        hdr_info = sndhdr.what(str(path))
        return isinstance(hdr_info, SndHeaders)
    
    if no_magic:
        return sndhdr_checker
    
    return magic_checker

is_audio = create_audio_checker(no_magic=no_magic)

def process_audio_file(
        path: str|Path,
        xmin:float = 0,
        xmax: float = None,
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
)->CandidateTracks:
    """Given the path to a single audio file, return a candidates track object.

    Args:
        path (str|Path): Path to the audio file
        xmin (float, optional): Start time to process the audio. Defaults to 0.
        xmax (float, optional): End tome for processing audio. If None, defaults to the
            maximum time. Defaults to None.
        min_max_formant (float, optional): The lowest max-formant value to try. 
            Defaults to 4000.
        max_max_formant (float, optional): The highest max formant to try. 
            Defaults to 7000.
        nstep (int, optional): The number of steps from the min to the max max formant. 
            Defaults to 20.
        n_formants (int, optional): The number of formants to track. Defaults to 4.
        window_length (float, optional): Window length of the formant analysis. 
            Defaults to 0.025.
        time_step (float, optional): Time step of the formant analyusis window. 
            Defaults to 0.002.
        pre_emphasis_from (float, optional): Pre-emphasis threshold. 
            Defaults to 50.
        smoother (Smoother, optional): The smoother method to use. 
            Defaults to `Smoother()`.
        loss_fun (Loss, optional): The loss function to use. 
            Defaults to Loss().
        agg_fun (Agg, optional): The loss aggregation function to use. 
            Defaults to Agg().

    Returns:
        (CandidateTracks): A `CandidateTracks` object to use.
    """
    if not is_audio(str(path)):
        raise TypeError(f"The file at {str(path)} is not an audio file")
    
    sound = pm.Sound(str(path))
    if not xmax:
        xmax = sound.xmax

    sound_to_process = sound.extract_part(from_time = xmin, to_time = xmax)
    candidates = CandidateTracks(
        samples=sound_to_process.values,
        sampling_frequency=sound_to_process.sampling_frequency,
        xmin = sound_to_process.xmin,
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

@delayed
def wrapped_audio(args_dict):
    return process_audio_file(**args_dict)

def process_directory(
        path: str|Path,
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
    """Given a path to a directoy of audio files, process them all.

    Args:
        path (str|Path): Path to the directory to process.
        min_max_formant (float, optional): The lowest max-formant value to try. 
            Defaults to 4000.
        max_max_formant (float, optional): The highest max formant to try. 
            Defaults to 7000.
        nstep (int, optional): The number of steps from the min to the max max formant. 
            Defaults to 20.
        n_formants (int, optional): The number of formants to track. Defaults to 4.
        window_length (float, optional): Window length of the formant analysis. 
            Defaults to 0.025.
        time_step (float, optional): Time step of the formant analyusis window. 
            Defaults to 0.002.
        pre_emphasis_from (float, optional): Pre-emphasis threshold. 
            Defaults to 50.
        smoother (Smoother, optional): The smoother method to use. 
            Defaults to `Smoother()`.
        loss_fun (Loss, optional): The loss function to use. 
            Defaults to Loss().
        agg_fun (Agg, optional): The loss aggregation function to use. 
            Defaults to Agg().

    Returns:
        (list[CandidateTracks]): A list of `CandidateTracks` objects.
    """
    if not isinstance(path, Path) and isinstance(path, str):
        path = Path(path)

    all_files = list(path.glob("*"))
    all_files = [x for x in all_files if x.is_file()]
    all_audio = [x for x in all_files if is_audio(x)]
    arg_list = [
            {"path": x,
            "min_max_formant": min_max_formant,
            "max_max_formant":max_max_formant,
            "nstep": nstep,
            "n_formants": n_formants,
            "window_length":window_length,
            "time_step":time_step,
            "pre_emphasis_from":pre_emphasis_from,
            "smoother":smoother,
            "loss_fun":loss_fun,
            "agg_fun":agg_fun
            }
            for x in all_audio
    ]
    n_jobs = cpu_count()

    all_candidates = Parallel(n_jobs=n_jobs)(
        wrapped_audio(args_dict=arg) for arg in tqdm(arg_list)
        )
    for x, path in zip(all_candidates, all_audio):
        x.file_name = Path(str(path)).name

    return all_candidates

