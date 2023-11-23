from fasttrackpy.tracks import CandidateTracks
from fasttrackpy.processors.outputs import write_data
from fasttrackpy.processors.aggs import Agg
from fasttrackpy.processors.smoothers import Smoother
from fasttrackpy.processors.losses import Loss
from fasttrackpy.patterns.just_audio import process_audio_file, \
                                            process_directory,\
                                            is_audio
from fasttrackpy.patterns.audio_textgrid import process_audio_textgrid
import parselmouth as pm
from pathlib import Path
from typing import Union

import click
import cloup
from cloup import Context, HelpFormatter, HelpTheme, Style

formatter_settings = HelpFormatter.settings(
    theme=HelpTheme(
        invoked_command=Style(fg='bright_yellow'),
        heading=Style(fg='bright_white', bold=True),
        constraint=Style(fg='magenta'),
        col1=Style(fg='green'),
    )
)

@cloup.group(show_subcommand_aliases=True)
def fasttrack():
    """jawn"""
    pass

@fasttrack.command(
    aliases = ["audio"],
    formatter_settings=formatter_settings,
    help="run fasttrack"
)
@cloup.option_group(
    "Audio Inputs",
    cloup.option(
        "--file", 
        type=click.Path(exists=True),
        help = "A single input audio file to process."
    ),
    cloup.option(
        "--dir", 
        type=click.Path(exists=True),
        help = "A directory of input audio files to process."
    ),
    help="Audio input file options",
    constraint=cloup.constraints.RequireAtLeast(1)
)
@cloup.option_group(
    "Audio Inputs",
    cloup.option(
        "--file", 
        type=click.Path(exists=True),
        help = "A single input audio file to process."
    ),
    cloup.option(
        "--dir", 
        type=click.Path(exists=True),
        help = "A directory of input audio files to process."
    ),
    help="Audio input file options",
    constraint=cloup.constraints.RequireAtLeast(1)
)
@cloup.option_group(
    "Output Destinations",
    cloup.option(
        "--output", 
        type=click.Path(),
        help = "Name of an output file",
    ),
    cloup.option(
        "--dest", 
        type=click.Path(),
        help = "Name of an output directory"
    ),
    help = "Output destination options",
    constraint=cloup.constraints.RequireAtLeast(1)    
)
@cloup.option_group(
    "Output Options",
    cloup.option(
        "--which-output", 
        type=click.Choice(["winner", "all"]), 
        default="winner",
        help = "Whether to save just the winner, or all candidates."\
               " Defaults to 'winner'"
    ),
    cloup.option(
        "--data-output", 
        type=click.Choice(["formants", "param"]), 
        default="formants",
        help = "Whether to save the formant data, "\
               "or smoothing parameter data."\
               " Defaults to 'formants'."   
    ),
    help = "Options for what data should be saved."
)

@cloup.option_group(
    "Audio processing",
    cloup.option(
        "--xmin", 
        type=click.FloatRange(min = 0), 
        default=0,
        help = "Start time for beginning analysis. "\
               "Defaults to 0(s). "\
               "(Ignored for multi-file input.)"
    ),
    cloup.option(
        "--xmax", 
        type=click.FloatRange(min = 0, min_open=True),
        help = "End time for analysis. "\
               "If not set, defaults to full duration. "\
               "(Ignored for multi-file input.)"
    ),
    cloup.option(
        "--min-max-formant", 
        type=click.FloatRange(min=0, min_open=True), 
        default=4000,
        help = "Start of possible max-formant range. Defaults to 4000(Hz)."
    ),
    cloup.option(
        "--max-max-formant", 
        type=click.FloatRange(min=0, min_open=True), 
        default=7000,
        help= "End of possible max-formant range. Defaults to 7000(Hz)."
    ),
    cloup.option(
        "--nstep", 
        type=click.IntRange(min=1), 
        default=20,
        help = "Number of max-formant steps to be evaluated. "\
               "Defaults to 20."
    ),
    cloup.option(
        "--n-formants", 
        type=click.IntRange(min=1), 
        default=4,
        help="Number of formants to track. Defaults to 4."
    ),
    cloup.option(
        "--window-length", 
        type=click.FloatRange(min = 0, min_open=True), 
        default=0.05,
        help = "Formant analysis window length. Defaults to 0.05(s)."
    ),
    cloup.option(
        "--time-step", 
        type=click.FloatRange(min = 0, min_open=True), 
        default=0.002,
        help = "Formant analysis window step size. Defaults to 0.002(s)."
    ),
    cloup.option(
        "--pre-emphasis-from", 
        type=click.FloatRange(min=0), 
        default=50,
        help="Pre-emphasis. Defaults to 50(Hz)."
    )    
)
@cloup.option_group(
    "Smoother options",
    cloup.option(
        "--smoother-method", 
        type=click.Choice(["dct_smooth", "dct_smooth_regression"]),
        default="dct_smooth",
        help="Smoother method to use. Defaults to 'dct_smooth' "\
             "(Discrete Cosine Transform)"
    ),
    cloup.option(
        "--smoother-order", 
        type=click.IntRange(min=1), 
        default=5,
        help = "Order of the smooth. Defaults to 5. (More is wigglier.)"
    ),
    cloup.option(
        "--loss-method", 
        type=click.Choice(["lmse", "mse"]), 
        default="lmse",
        help = "The loss function comparing formants to smoothed tracks. "\
               "Defaults to lmse (log mean squared error)."
    )
)
def audio(
        file: Union[str, Path] = None,
        dir: Union[str,Path] = None,
        output: Union[str, Path] = None,
        dest: Union[str, Path] = None,
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
    """Run fasttrack.

    Args:
        file (Union[str, Path], optional): A single input audio file to process. 
            Defaults to None.
        dir (Union[str,Path], optional): A directory of input audio files to process. Defaults to None.
        output (Union[str, Path], optional): Name of an output file. Defaults to None.
        dest (Union[str, Path], optional): "Name of an output directory. Defaults to None.
        which_output (str, optional): Whether to save just the winner, 
            or all candidates. Defaults to 'winner'. Defaults to "winner".
        data_output (str, optional): Whether to save the formant data,
            or smoothing parameter data.
            Defaults to "formants".
        smoother_method (str, optional): Smoother method to use. Defaults to 'dct_smooth'
            (Discrete Cosine Transform)
        smoother_order (int, optional): Order of the smooth. 
            Defaults to 5. (More is wigglier.)
        loss_method (str, optional): The loss function comparing formants to 
            smoothed tracks.
            Defaults to lmse (log mean squared error).
        xmin (float, optional): Start time for beginning analysis.
            Defaults to 0(s). 
            Defaults to 0.
        xmax (float, optional): End time for analysis.
            If not set, defaults to full duration.
            Defaults to None.
        min_max_formant (float, optional): Start of possible max-formant range. 
            Defaults to 4000(Hz).
        max_max_formant (float, optional): End of possible max-formant range. 
            Defaults to 7000(Hz).
        nstep (int, optional): Number of max-formant steps to be evaluated.
            Defaults to 20.
        n_formants (int, optional): Number of formants to track.
            Defaults to 4.
        window_length (float, optional): Formant analysis window length. 
            Defaults to 0.05(s).
        time_step (float, optional): Formant analysis window step size.
            Defaults to 0.002(s)
        pre_emphasis_from (float, optional): Pre-emphasis. Defaults to 50(Hz)
    """
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
                   destination=dest,
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
            destination=dest,
            which = which_output,
            output=data_output
            ) for x in candidate_list]
        
if __name__ == "__main__":
    fasttrack()





# # @cloup.option_group(
# #     "Textgrid",
# #     cloup.option(
# #         "--textgrid",
# #         type = click.Path(exists=True),
# #         help = "A textgrid to process"
# #     ),
# #     cloup.option(
# #         "--entry-classes",
# #         default="SequenceInterval",
# #         type = click.STRING,
# #         help = "Entry classes for TextGrid"
# #     ),
# #     cloup.option(
# #         "--target-tier",
# #         type = click.STRING,
# #         help = "The tier to target"        
# #     ),
# #     click.option(
# #         "--target-labels",
# #         default = ".",
# #         type = click.STRING,
# #         help = "Regex for target labels"
# #     )
# # )