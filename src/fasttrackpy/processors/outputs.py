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

    smooth_df = pl.DataFrame(
        data = self.smoothed_formants.T,
        schema=smooth_names
    )

    out_df = pl.concat([orig_df, smooth_df], how = "horizontal")
    
    out_df = out_df.with_columns(
        time = pl.lit(self.time_domain),
        max_formant = pl.lit(self.maximum_formant),
        n_formant = pl.lit(self.n_formants),
        smooth_method = pl.lit(self.smoother.smooth_fun.__name__)
    )

    return out_df