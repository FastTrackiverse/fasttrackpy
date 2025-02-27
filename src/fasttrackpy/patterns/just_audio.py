import warnings
from pathlib import Path
from typing import Union
from collections.abc import Callable
import parselmouth as pm
from fasttrackpy import CandidateTracks,\
                        Smoother,\
                        Loss,\
                        Agg
from fasttrackpy.processors.heuristic import MinMaxHeuristic, SpacingHeuristic
from fasttrackpy.utils.safely import safely, filter_nones

import filetype

from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)



def is_audio(path: str|Path)->bool:
    """Checks whether a file is an audio file

    Args:
        path (str): Path to the file in question

    Returns:
        (bool): Whether or not the file is an audio file
    """    
    info = filetype.guess(path)
    if info is not None:
        return "audio" in info.mime
    return False

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
        agg_fun: Agg = Agg(),
        heuristics: list[MinMaxHeuristic|SpacingHeuristic] = []
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
        heuristics (list[MinMaxHeuristic|SpacingHeuristic]):
            A list of formant tracking heuristics to use.            

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
        agg_fun=agg_fun,
        heuristics=heuristics
    )
    candidates.file_name = Path(str(path)).name
    return candidates

@delayed
@safely(message = "There was a problem processing an audio file.")
def get_candidates_delayed(args_dict):
    return process_audio_file(**args_dict)

@safely(message = "There was a problem processing an audio file.")
def get_candidates(args_dict):
    return process_audio_file(**args_dict)

def run_candidates(arg_list, parallel:bool):
    if parallel:
        n_jobs = cpu_count()
        all_candidates = Parallel(n_jobs=n_jobs)(
            get_candidates_delayed(args_dict=arg) for arg in tqdm(arg_list)
            )
        return all_candidates
    
    all_candidates = [get_candidates(args_dict=arg) for arg in tqdm(arg_list)]
    return all_candidates

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
        agg_fun: Agg = Agg(),
        heuristics: list[MinMaxHeuristic|SpacingHeuristic] = []
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
        heuristics (list[MinMaxHeuristic|SpacingHeuristic]):
            A list of formant tracking heuristics to use.            

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
            "agg_fun":agg_fun,
            "heuristics":heuristics
            }
            for x in all_audio
    ]

    windows_3_12 = os.name != "posix" and \
            sys.version_info.major == 3 and \
            sys.version_info.minor == 12

    all_candidates = run_candidates(
        arg_list, not windows_3_12
    )

    all_candidates, all_audio = filter_nones(all_candidates, [all_candidates, all_audio])

    for x, path in zip(all_candidates, all_audio):
        x.file_name = Path(str(path)).name

    return all_candidates

