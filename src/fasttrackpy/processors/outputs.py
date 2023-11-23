import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as mp

ptolmap = {"F1" :"#4477AA", "F2": "#EE6677", "F3": "#228833", "F4": "#CCBB44"}


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


def spectrogram(
        self, 
        formants = 3, 
        maximum_frequency=3500, 
        tracks = True, 
        dynamic_range=60, 
        figsize = (8,5)
    ):

    spctgrm = self.sound.to_spectrogram(
        maximum_frequency=maximum_frequency
    )
    Time, Hz = spctgrm.x_grid(), spctgrm.y_grid()
    db = 10 * np.log10(spctgrm.values)
    min_shown = db.max() - dynamic_range
    n_time_steps = len(self.formants[0])
    point_times = [0.025 + time_step*0.002 for time_step in range(n_time_steps)]    
    
    mp.figure(figsize=figsize)
    mp.pcolormesh(Time, Hz, db, vmin=min_shown, cmap='Greys')
    mp.ylim([spctgrm.ymin, spctgrm.ymax])
    mp.xlabel("Time (s)")
    mp.ylabel("Frequency (Hz)")

    formant_cols = [f"F{x+1}" for x in range(formants)]
    data = self.to_df()
    formant_cols = [x for x in formant_cols if x in data.columns]

    data = data\
        .select(["time"]+formant_cols)\
        .melt(id_vars = "time")\
        .with_columns(
            pl.col("variable")\
            .map_dict(remapping=ptolmap)\
            .alias("color")
        )
    
    if tracks:
        mp.scatter(x = "time", y="value", c="color", data=data, marker = '.')    

        mp.scatter (point_times[0:len(self.formants[0]):3], 
                    self.smoothed_formants[0][0:len(self.formants[0]):3], c="red", marker="+")
        mp.scatter (point_times[0:len(self.formants[1]):3], 
                    self.smoothed_formants[1][0:len(self.formants[1]):3], c="blue", marker="+")
        mp.scatter (point_times[0:len(self.formants[2]):3], 
                    self.smoothed_formants[2][0:len(self.formants[2]):3], c="green", marker="+")

        if formants == 4:
            mp.scatter (point_times[0:len(self.formants[3]):3], 
                        self.smoothed_formants[3][0:len(self.formants[3]):3], c="darkturquoise", marker="+")

def candidate_spectrograms(
        self, 
        formants = 3, 
        maximum_frequency = 3500, 
        dynamic_range=60,
        figsize=(12,8)
    ):
    
    spectrogram = self.sound.to_spectrogram(
        maximum_frequency=maximum_frequency,
        time_step=0.005
        )
    Time = spectrogram.x_grid()
    Hz = spectrogram.y_grid()

    db = 10 * np.log10(spectrogram.values)
    min_shown = db.max() - dynamic_range
    n_time_steps = len(self.candidates[0].formants[0])
    point_times = [0.025 + time_step*0.002 for time_step in range(n_time_steps)]    

    # for plotting layout    
    dims = np.array([4, self.nstep//4])
    panel_columns = dims.max()
    panel_rows = dims.min()
    
    fig = mp.figure(figsize=figsize)
    gs = fig.add_gridspec(panel_rows,panel_columns, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex='col', sharey='row')

    formant_cols = [f"F{x+1}" for x in range(formants)]
    for i in range (panel_rows):
        for j in range(panel_columns):
            analysis = i*panel_columns+j

            if analysis == self.winner_idx:
                axs[i, j].pcolormesh(Time, Hz, db, vmin=min_shown, cmap='magma')
            else:
                axs[i, j].pcolormesh(Time, Hz, db, vmin=min_shown, cmap='binary')

            axs[i, j].set_ylim([0, spectrogram.ymax])

            data = self.candidates[analysis].to_df()
            formant_cols = [x for x in formant_cols if x in data.columns]

            data = data\
                .select(["time"]+formant_cols)\
                .melt(id_vars = "time")\
                .with_columns(
                    pl.col("variable")\
                    .map_dict(remapping=ptolmap)\
                    .alias("color")
                )
            
            axs[i,j].scatter(
                x = "time",
                y = "value",
                c = "color",
                data = data,
                marker = "."
            )

            axs[i,j].text(
                x = 0.1,
                y = spectrogram.ymax * 0.9,
                #s = str(analysis)
                s = str(round(self.candidates[analysis].maximum_formant))
            )
            

            axs[i, j].scatter (point_times[0:len(self.candidates[analysis].formants[0]):3], 
                               self.candidates[analysis].smoothed_formants[0][0:len(self.candidates[analysis].formants[0]):3], 
                               c="red", marker="+", s = 5)
            axs[i, j].scatter (point_times[0:len(self.candidates[analysis].formants[1]):3], 
                               self.candidates[analysis].smoothed_formants[1][0:len(self.candidates[analysis].formants[1]):3], 
                               c="blue", marker="+", s = 5)
            axs[i, j].scatter (point_times[0:len(self.candidates[analysis].formants[2]):3], 
                               self.candidates[analysis].smoothed_formants[2][0:len(self.candidates[analysis].formants[2]):3], 
                               c="green", marker="+", s = 5)

            if formants == 4:
                axs[i, j].scatter (point_times[0:len(self.candidates[analysis].formants[3]):3], 
                               self.candidates[analysis].smoothed_formants[3][0:len(self.candidates[analysis].formants[3]):3], 
                               c="darkturquoise", marker="+", s = 5)    
                
    for ax in fig.get_axes():
        ax.label_outer()