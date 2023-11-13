import numpy as np
import polars as pl
from pathlib import Path

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

    if self.file_name:
        out_df = out_df.with_columns(
            file_name = pl.lit(self.file_name)
        )

    if self.id:
        out_df = out_df.with_columns(
            id = pl.lit(self.id)
        )        

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

    if self.file_name:
        param_df = param_df.with_columns(
            file_name = pl.lit(self.file_name)
        )

    if self.id:
        param_df = param_df.with_columns(
            id = pl.lit(self.id)
        )        

    return param_df

def write_winner(
        candidates,
        file: Path = None,
        destination: Path = None,
        output: str = "formants"
):
    df = candidates.winner.to_df(output=output)
    if file:
        df.write_csv(file = file)
        return
    
    if destination and candidates.winner.file_name:
        file = destination.joinpath(
            candidates.winner.file_name
        ).with_suffix(".csv")
        df.write_csv(file = file)
        return

    if destination:
        file = destination.joinpath("output.csv")
        df.write_csv(file = file)
    
    raise ValueError("Either 'file' or 'destination' needs to be set")
