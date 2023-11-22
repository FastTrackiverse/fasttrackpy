import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as mp


def add_metadata(self, out_df):
    if self.file_name:
        out_df = out_df.with_columns(
            file_name = pl.lit(self.file_name)
        )

    if self.id:
        out_df = out_df.with_columns(
            id = pl.lit(self.id)
        )
    return out_df

def formant_to_dataframe(self):
    """Return data as a data frame

    Returns:
        (pl.DataFrame): A data frame
    """
    orig_names = [
        f"F{x}" for x in np.arange(self.n_measured_formants)+1
    ]
    smooth_names = [
        f"F{x}_s" for x in np.arange(self.n_measured_formants)+1
    ]

    orig_df = pl.DataFrame(
        data = self.formants[0:self.n_measured_formants].T,
        schema=orig_names
    )

    smooth_df = pl.DataFrame(
        data = self.smoothed_formants[0:self.n_measured_formants].T,
        schema=smooth_names
    )

    out_df = pl.concat([orig_df, smooth_df], how = "horizontal")
    
    out_df = out_df.with_columns(
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
        data = self.parameters.T,schema=schema
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
        output: str = "formants"
):
    df = candidates.to_df(which = which, output = output)
    if file:
        df.write_csv(file = file)
        return
    
    if destination and candidates.file_name:
        if not isinstance(destination, Path):
            destination = Path(destination)
        file = destination.joinpath(
            candidates.winner.file_name
        ).with_suffix(".csv")
        df.write_csv(file = file)
        return

    if destination:
        file = destination.joinpath("output.csv")
        df.write_csv(file = file)
        return
    
    raise ValueError("Either 'file' or 'destination' needs to be set")


def spectrogram(self, formants = 3, maximum_frequency=3500, tracks = True, dynamic_range=60, figsize = (8,5)):

    spctgrm = self.sound.to_spectrogram(maximum_frequency=maximum_frequency)
    Time, Hz = spctgrm.x_grid(), spctgrm.y_grid()
    db = 10 * np.log10(spctgrm.values)
    min_shown = db.max() - dynamic_range
    n_time_steps = len(self.formants[0])
    point_times = [0.025 + time_step*0.002 for time_step in range(n_time_steps)]    
    
    mp.figure(figsize=figsize)
    mp.pcolormesh(Time, Hz, db, vmin=min_shown, cmap='magma')
    mp.ylim([spctgrm.ymin, spctgrm.ymax])
    mp.xlabel("Time (s)")
    mp.ylabel("Frequency (Hz)")
    
    if tracks:
        mp.scatter (point_times, self.formants[0], c="red")
        mp.scatter (point_times, self.formants[1], c="blue")
        mp.scatter (point_times, self.formants[2], c="green")
        if formants == 4:
            mp.scatter (point_times, self.formants[3], c="darkturquoise")    

def candidate_spectrograms(self, formants = 3, maximum_frequency = 3500, dynamic_range=60,figsize=(12,8)):
    
    spectrogram = self.sound.to_spectrogram(maximum_frequency=maximum_frequency,time_step=0.005)
    Time, Hz = spectrogram.x_grid(), spectrogram.y_grid()
    db = 10 * np.log10(spectrogram.values)
    min_shown = db.max() - dynamic_range
    n_time_steps = len(self.candidates[0].formants[0])
    point_times = [0.025 + time_step*0.002 for time_step in range(n_time_steps)]    
    
    # for plotting layout
    match self.nstep:
        case 8:
            panel_columns = 4
            panel_rows = 2
        case 12:
            panel_columns = 4
            panel_rows = 3
        case 16:
            panel_columns = 4
            panel_rows = 4
        case 20:
            panel_columns = 5
            panel_rows = 4
        case 24:
            panel_columns = 6
            panel_rows = 4
    
    fig = mp.figure(figsize=figsize)
    gs = fig.add_gridspec(panel_rows,panel_columns, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex='col', sharey='row')

    #gs = fig.add_gridspec(3, hspace=0)
    #axs = gs.subplots(sharex=True, sharey=True)

    for i in range (panel_rows):
        for j in range(panel_columns):
            axs[i, j].pcolormesh(Time, Hz, db, vmin=min_shown, cmap='magma')
            axs[i, j].set_ylim([0, spectrogram.ymax])
            analysis = i*3+j
            axs[i, j].scatter (point_times, self.candidates[analysis].formants[0], c="red", s = 5)
            axs[i, j].scatter (point_times, self.candidates[analysis].formants[1], c="blue", s = 5)
            axs[i, j].scatter (point_times, self.candidates[analysis].formants[2], c="green", s = 5)    
            if formants == 4:
                axs[i, j].scatter (point_times, self.candidates[analysis].formants[3], c="darkturquoise", s = 5)    

    for ax in fig.get_axes():
        ax.label_outer()