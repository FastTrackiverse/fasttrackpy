import numpy as np
import polars as pl

def to_dataframe(self):
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
        data = self.formants.T,
        schema=orig_names
    )
    orig_df = orig_df.with_columns(
        time = pl.lit(self.time_domain)
    )

    smooth_df = pl.DataFrame(
        data = self.smoothed_formants.T,
        schema=smooth_names
    )
    out_df = pl.concat([orig_df, smooth_df], how = "horizontal")

    return out_df