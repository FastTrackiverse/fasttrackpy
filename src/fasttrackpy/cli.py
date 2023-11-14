from fasttrackpy.tracks import CandidateTracks
from fasttrackpy.processors.outputs import write_data
from fasttrackpy.processors.aggs import Agg
from fasttrackpy.processors.smoothers import Smoother
from fasttrackpy.processors.losses import Loss
from fasttrackpy.patterns.just_audio import process_audio_file, \
                                            process_directory,\
                                            is_audio
import parselmouth as pm
from pathlib import Path
from typing import Union
import click
import cloup

@cloup.command()
@cloup.option_group(
    "Inputs",
    cloup.option("--file", type=click.Path(exists=True)),
    cloup.option("--dir", type=click.Path(exists=True)),
    help="Input file options",
    constraint=cloup.constraints.RequireAtLeast(1)
)
@cloup.option_group(
    "Output Destinations",
    cloup.option("--output", type=click.Path(),
                 help = "Name of an output file"),
    cloup.option("--destination", type=click.Path(),
                 help = "Name of an output directory"),
    help = "Output destination options",
    constraint=cloup.constraints.RequireAtLeast(1)    
)
@cloup.option_group(
    "Output Options",
    cloup.option("--which-output", type=click.Choice(["winner", "all"]), default="winner"),
    cloup.option("--data-output", type=click.Choice(["formants", "param"]), default="formants")
)
@cloup.option_group(
    "Audio processing",
    cloup.option("--xmin", type=float, default=0),
    cloup.option("--xmax", type=float),
    cloup.option("--min-max-formant", type=float, default=4000),
    cloup.option("--max-max-formant", type=float, default=7000),
    cloup.option("--nstep", type=int, default=20),
    cloup.option("--n-formants", type=int, default=4),
    cloup.option("--window-length", type=float, default=0.05),
    cloup.option("--time-step", type=float, default=0.002),
    cloup.option("--pre-emphasis-from", type=float, default=50)    
)
@cloup.option_group(
    "Smoother options",
    cloup.option("--smoother-method", type=str, default="dct_smooth"),
    cloup.option("--smoother-order", type=int, default=5),
    cloup.option("--loss-method", type=str, default="lmse"),
)
def fasttrack(
        file: Union[str, Path] = None,
        dir: Union[str,Path] = None,
        output: Union[str, Path] = None,
        destination: Union[str, Path] = None,
        which_output: str = "winner",
        data_output: str = "formants",
        smoother_method: str = "dct_smooth",
        smoother_order: int = 5,
        loss_method: str = "lmse",
        xmin:float = 0,
        xmax: float = None,
        min_max_formant:float = 4000,
        max_max_formant:float = 7000,
        nstep:int = 20,
        n_formants: int = 4,
        window_length: float = 0.05,
        time_step: float = 0.002,
        pre_emphasis_from: float = 50
):
    
    smoother_kwargs = {
        "method": smoother_method,
        "order": smoother_order
    }

    loss_kwargs = {
        "method": loss_method
    }

    smoother = Smoother(**smoother_kwargs)
    loss_fun = Loss(**loss_kwargs)
    agg_fun = Agg()

    if file and is_audio(file):

        candidates = process_audio_file(
            path = file,
            xmin = xmin,
            xmax = xmax,
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

        write_data(candidates=candidates, 
                   file=output, 
                   which=which_output, 
                   output=data_output
        )
    if dir:
        candidate_list = process_directory(
            path = dir,
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

        [write_data(
            x, 
            destination=destination,
            which = which_output,
            output=data_output
            ) for x in candidate_list]