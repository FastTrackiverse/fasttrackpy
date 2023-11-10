import parselmouth as pm
import numpy as np
from fasttrackpy.processors.smoothers import Smoother
from fasttrackpy.processors import losses
from fasttrackpy.processors import aggs

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import polars as pl

from typing import Union

class Track:
    def __init__(
            self,
            sound: pm.Sound,
            n_formants: int = 4,
            window_length: float = 0.05,
            time_step: float = 0.002,
            pre_emphasis_from: float = 50,
            smoother:Smoother = Smoother(),
            loss_fun = losses.lmse,
            agg_fun = aggs.agg_sum
    ):
        self.sound = sound
        self.n_formants = n_formants
        self.window_length = window_length
        self.time_step = time_step
        self.pre_emphasis_from = pre_emphasis_from
        self.smoother = smoother
        self.loss_fun = loss_fun
        self.agg_fun = agg_fun

class OneTrack(Track):
    def __init__(
            self,
            maximum_formant: float,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.maximum_formant = maximum_formant

        self.formants, self.time_domain = self._track_formants()
        self.n_measured_formants = self._get_measured()
        self.imputed_formants = self._impute_formants()
        self.smoothed_list = self._smooth_formants()
    
    def __repr__(self):
        return f"A formant track object. {self.formants.shape}"

    def _track_formants(self):
        formant_obj = self.sound.to_formant_burg(
            time_step = self.time_step,
            max_number_of_formants = self.n_formants,
            maximum_formant = self.maximum_formant,
            window_length = self.window_length,
            pre_emphasis_from = self.pre_emphasis_from
        )

        time_domain = formant_obj.xs()
        tracks = np.array(
            [
                [
                    formant_obj.get_value_at_time(i+1, x)
                    for x in time_domain
                ]
                for i in range(int(np.floor(self.n_formants)))
            ]
        )

        return tracks, time_domain

    def _smooth_formants(self):
        smoothed_list = [
          self.smoother.smooth(x) 
            for x in self.imputed_formants
        ]
    
        return smoothed_list
    
    def _get_measured(self):
        nan_tracks = np.isnan(self.formants)
        all_nan = np.all(nan_tracks, axis = 1)
        if np.any(all_nan):
            return np.argmax(all_nan)
        return all_nan.shape[0]
    
    def _impute_formants(self):
        imp = IterativeImputer(max_iter=10, random_state=0)
        to_impute = self.formants[0:self.n_measured_formants,:]
        nan_entries = np.isnan(to_impute)
        if not np.any(nan_entries):
            return to_impute

        imp.fit(np.transpose(to_impute))
        imputed = np.transpose(
            imp.transform(np.transpose(to_impute))
        )
        return imputed

    @property
    def smoothed_formants(self):
        return np.array(
            [x.smoothed for x in self.smoothed_list]
        )

    @property
    def smooth_error(self):
        msqe =  self.loss_fun(
            self.formants[0:self.n_measured_formants], 
            self.smoothed_formants[0:self.n_measured_formants]
        )
        error = self.agg_fun(msqe)
        return error
    
    def to_dataframe(self):
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

class CandidateTracks(Track):
    def __init__(
        self,
        min_max_formant: float = 5000,
        max_max_formant: float = 7000,
        nstep = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_max_formant = min_max_formant
        self.max_max_formant = max_max_formant
        self.nstep = nstep
        self.max_formants = np.linspace(
            start = self.min_max_formant,
            stop = self.max_max_formant,
            num = self.nstep
        )

        self.candidates = [
            OneTrack(
                maximum_formant=x,
                **kwargs
            ) for x in self.max_formants
        ]

        self.min_n_measured = np.array([
            x.n_measured_formants 
            for x in self.candidates
        ]).min()

        self._normalize_n_measured()

        self.smooth_errors = np.array(
            [x.smooth_error for x in self.candidates]
        )

        self.winner_idx = np.argmin(self.smooth_errors)
        self.winner = self.candidates[self.winner_idx]

    def _normalize_n_measured(self):
        for track in self.candidates:
            track.n_measured_formants = self.min_n_measured