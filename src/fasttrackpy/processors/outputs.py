import numpy as np
import pickle
import parselmouth as pm
import polars as pl
from aligned_textgrid import SequenceInterval
from pathlib import Path
import matplotlib.pyplot as mp
import copy
import logging

ptolmap = {"F1" :"#4477AA",
           "F1_s": "#4477AA",
           "F2": "#EE6677",
           "F2_s": "#EE6677",
           "F3": "#228833",
           "F3_s": "#228833",
           "F4": "#CCBB44",
           "F4_s": "#CCBB44"}

def add_metadata(self, out_df):
    if self.file_name:
        out_df = out_df.with_columns(
            file_name = pl.lit(self.file_name)
        )

    if self.id:
        out_df = out_df.with_columns(
            id = pl.lit(self.id)
        )

    if self.group:
        out_df = out_df.with_columns(
            group = pl.lit(self.group)
        )

    if isinstance(self.interval, SequenceInterval) :
        out_df = out_df.with_columns(
            label = pl.lit(self.interval.label)
        )

    return out_df

def formant_to_dataframe(self):
    """Return data as a data frame

    Returns:
        (pl.DataFrame): A data frame
    """
    orig_names = [
        f"F{x}" for x in np.arange(self.n_formants)+1
    ]
    smooth_names = [
        f"F{x}_s" for x in np.arange(self.n_formants)+1
    ]

    orig_df = pl.DataFrame(
        data = self.formants[0:self.n_formants],
        schema=orig_names
    )

    smooth_df = pl.DataFrame(
        data = self.smoothed_formants[0:self.n_formants],
        schema=smooth_names
    )

    out_df = pl.concat([orig_df, smooth_df], how = "horizontal")

    out_df = out_df.with_columns(
        error = pl.lit(self.smooth_error),
        time = pl.lit(self.time_domain),
        max_formant = pl.lit(self.maximum_formant),
        n_formant = pl.lit(self.n_formants),
        smooth_method = pl.lit(self.smoother.smooth_fun.__name__)
    )

    out_df = add_metadata(self, out_df)

    return out_df

def param_to_dataframe(self):
    """Return data as a data frame

    Returns:
        (pl.DataFrame): A data frame
    """

    schema = [
        f"F{x}" for x in
        np.arange(self.parameters.shape[0])+1
    ]
    param_df = pl.DataFrame(
        data = self.parameters,schema=schema
    )

    param_df = param_df.with_columns(
        error = pl.lit(self.smooth_error)
    )

    param_df = add_metadata(self, param_df)

    return param_df

def get_big_df(self, output):
        all_df = [x.to_df(output = output) for x in self.candidates]
        all_df = [
            x.with_columns(
                candidate = idx+1
            )
            for idx, x in enumerate(all_df)
        ]

        big_df = pl.concat(all_df, how = "diagonal")
        return big_df

def write_data(
        candidates,
        file: Path = None,
        destination: Path = None,
        which: str = "winner",
        output: str = "formants",
        separate: bool = False
):
    if destination and not isinstance(destination, Path):
        destination = Path(destination)

    if file and not isinstance(file, Path):
        file = Path(file)

    if type(candidates) is list:
        df = pl.concat(
            [x.to_df(which = which, output = output) for x in candidates],
            how = "diagonal"
        )
    else:
        df = candidates.to_df(which = which, output = output)

    if file:
        df.write_csv(file = str(file.resolve()))
        return

    if destination and "file_name" in df.columns and not separate:
        file = destination.joinpath(
            df["file_name"][0]
        ).with_suffix(".csv")
        df.write_csv(file = str(file.resolve()))
        return

    if destination and "file_name" in df.columns:
        unique_entries = df \
            .select("file_name", "group")\
            .unique()\
            .with_columns(
                pl.concat_str(
                    [pl.col("file_name"),
                     pl.col("group")],
                     separator="_"
                ).alias("newname")
            ) \
            .rows_by_key("newname", named = True)

        for newname in unique_entries:
            out_df = df \
                .filter(
                    (pl.col("file_name") == unique_entries[newname][0]["file_name"]) &
                    (pl.col("group") == unique_entries[newname][0]["group"])
                )

            out_df.write_csv(
                file = str(destination.joinpath(newname).with_suffix(".csv").resolve())
            )
        return

    if destination:
        file = destination.joinpath("output.csv")
        df.write_csv(file = str(file.resolve()))
        return

    raise ValueError("Either 'file' or 'destination' needs to be set")


def spectrogram(
        self,
        formants:int = 3,
        maximum_frequency:float = 3500,
        tracks:bool = True,
        dynamic_range:float =60,
        figsize: tuple[float, float] = (8,5),
        color_scale: str ="Greys",
        file_name:Path|None = None,
        dpi:float = 100
    ):
    """
    This will plot the spectrogram and formant tracks
    of a single candidate track. If a `file_name` is 
    provided, it will save the plot to disk.

    Args:
        formants (int, optional): 
            The number of formants to plot. 
            Defaults to 3.
        maximum_frequency (float, optional):
            The frequency range the spectrogram and formants
            will be plotted up to. Defaults to 3500.
        tracks (bool, optional): 
            Whether or not to plot the formant tracks. 
            Defaults to True. If False, just the spectogram
            will be plotted.
        dynamic_range (float, optional): 
            A all spectrogram values below the dynamic range. 
            will be plotted as white. Defaults to 60.
        figsize (tuple[float, float], optional):
            Width and height of the figure in inches. 
            Defaults to (8,5).
        color_scale (str, optional): 
            A named matplotlib color scale for the spectrogram.
            Defaults to "Greys". 
            See [here](https://matplotlib.org/stable/users/explain/colors/colormaps.html) 
            for more options.
        file_name (Path | None, optional):
            If provided, how to save the spectrogram. If not provided (None) 
            the plot will show interactively. Defaults to None.
        dpi (float, optional):
            If the plot is being saved, its image resolution in 
            dots per inch. Defaults to 100.
    """

    spctgrm = self.sound.to_spectrogram(
        maximum_frequency=maximum_frequency
    )
    Time, Hz = spctgrm.x_grid(), spctgrm.y_grid()
    db = 10 * np.log10(spctgrm.values)
    min_shown = db.max() - dynamic_range

    mp.figure(figsize=figsize)
    mp.pcolormesh(Time, Hz, db, vmin=min_shown, cmap=color_scale)
    mp.ylim([spctgrm.ymin, spctgrm.ymax])
    mp.xlabel("Time (s)")
    mp.ylabel("Frequency (Hz)")

    formant_cols = [f"F{x+1}" for x in range(formants)]
    smooth_cols = [f"F{x+1}_s" for x in range(formants)]
    data = self.to_df()
    all_cols = [x for x in formant_cols+smooth_cols if x in data.columns]

    data = data\
        .select(["time"]+all_cols)\
        .melt(id_vars = "time")\
        .with_columns(
            pl.col("variable")\
            .replace(ptolmap)\
            .alias("color")
            )

    if tracks:
        mp.scatter(x = "time",
                   y="value",
                   c="color",
                   marker = ".",
                   data=data.filter(
                       ~pl.col("variable").str.contains("_s")
                   ))
        mp.scatter(x = "time",
                   y="value",
                   c="color",
                   marker = "+",
                   data=data.filter(
                       pl.col("variable").str.contains("_s")
                   ))

    if file_name:
        mp.savefig(file_name, dpi=dpi)
        mp.close()


def candidate_spectrograms(
        self,
        formants:int = 3,
        maximum_frequency:float = 3500,
        dynamic_range:float=60,
        figsize:tuple[float, float]=(12,8),
        file_name: Path|None = None,
        dpi:float = 75
    ):
    """ 
    This will plot a grid of the candidate formant
    tracks and their spectrograms. If a `file_name`
    is provided, it will save the plot to disk.

    Args:
        formants (int, optional): 
            The number of formants to plot. 
            Defaults to 3.
        maximum_frequency (float, optional):
            The frequency range the spectrogram and formants
            will be plotted up to. Defaults to 3500.
        dynamic_range (float, optional): 
            A all spectrogram values below the dynamic range. 
            will be plotted as white. Defaults to 60.
        figsize (tuple[float, float], optional):
            Width and height of the figure in inches. 
            Defaults to (8,5).
        file_name (Path | None, optional):
            If provided, how to save the spectrogram. If not provided (None) 
            the plot will show interactively. Defaults to None.
        dpi (float, optional):
            If the plot is being saved, its image resolution in 
            dots per inch. Defaults to 75
    """

    spectrogram = self.sound.to_spectrogram(
        maximum_frequency=maximum_frequency,
        time_step=0.005
        )
    Time = spectrogram.x_grid()
    Hz = spectrogram.y_grid()

    db = 10 * np.log10(spectrogram.values)
    min_shown = db.max() - dynamic_range

    # for plotting layout
    dims = np.array([4, self.nstep//4])
    panel_columns = dims.max()
    panel_rows = dims.min()

    fig = mp.figure(figsize=figsize)
    gs = fig.add_gridspec(panel_rows,panel_columns, hspace=0.18, wspace=0.05)
    axs = gs.subplots(sharex='col', sharey='row')

    formant_cols = [f"F{x+1}" for x in range(formants)]
    smooth_cols = [f"F{x+1}_s" for x in range(formants)]

    for i in range (panel_rows):
        for j in range(panel_columns):
            analysis = i*panel_columns+j

            if analysis == self.winner_idx:
                axs[i, j].pcolormesh(Time, Hz, db, vmin=min_shown, cmap='jet')
            else:
                axs[i, j].pcolormesh(Time, Hz, db, vmin=min_shown, cmap='binary')

            axs[i, j].set_ylim([0, spectrogram.ymax])

            data = self.candidates[analysis].to_df()
            all_cols = [x for x in formant_cols+smooth_cols if x in data.columns]

            data = data\
                .select(["time"]+all_cols)\
                .melt(id_vars = "time")\
                .with_columns(
                    pl.col("variable")\
                    .replace(ptolmap)\
                    .alias("color")
                )

            axs[i,j].scatter(
                x = "time",
                y = "value",
                c = "color",
                data=data.filter(
                       ~pl.col("variable").str.contains("_s")
                ),
                s = 5,
                marker = "."
            )
            axs[i,j].scatter(
                x = "time",
                y = "value",
                c = "color",
                data = data.filter(
                    pl.col("variable").str.contains("_s")
                ),
                s = 5,
                marker = "+"
            )

            axs[i, j].set_title(str(round(self.candidates[analysis].maximum_formant)),y=0.95)

    if file_name:
        mp.savefig(file_name, dpi=dpi, bbox_inches='tight')
        mp.close()

    for ax in fig.get_axes():
        ax.label_outer()


def pickle_candidates(
    candidates,
    file: Path|str
    ):
    """
    This will save a CandidateTracks object to
    disk as a "pickle" file. Due to the way sound 
    objects are handled, only use `unpickle_candidates()`
    to re-load the pickle object.

    Args:
        candidates (CandidateTracks):
            A CandidateTracks object to pickle.
        file (Path | str): 
            The file location to save the pickle file
            to.
    """
    if type(file) is str:
        file = Path(file)

    tmp_candidates = copy.deepcopy (candidates)

    with file.open('wb') as f:
        pickle.dump(tmp_candidates, f)


def unpickle_candidates(
        file: Path|str
        ) :
    """
    This will load a CandidateTracks object 
    that was pickled with `pickle_candidates()`.

    Args:
        file (Path | str):
            The pickled CandidateTracks object
            to unpickle.

    Returns:
        (CandidateTracks):
            A CandidateTracks object.
    """
    if type(file) is str:
        file = Path(file)    

    with file.open('rb') as f:
        candidates = pickle.load(f)

    return candidates
