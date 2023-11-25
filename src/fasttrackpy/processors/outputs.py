import numpy as np
import polars as pl
from aligned_textgrid import SequenceInterval
from pathlib import Path
import logging

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
        data = self.parameters.T,schema=schema
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
        output: str = "formants"
):
    if type(candidates) is list:
        df = pl.concat(
            [x.to_df(which = which, output = output) for x in candidates],
            how = "diagonal"
        )
    else:
        df = candidates.to_df(which = which, output = output)

    if file:
        df.write_csv(file = file)
        return
    
    if destination and "file_name" in df.columns:
        if not isinstance(destination, Path):
            destination = Path(destination)
        file = destination.joinpath(
            df["file_name"][0]
        ).with_suffix(".csv")
        df.write_csv(file = file)
        return

    if destination:
        file = destination.joinpath("output.csv")
        df.write_csv(file = file)
        return
    
    raise ValueError("Either 'file' or 'destination' needs to be set")
