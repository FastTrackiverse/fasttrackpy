import parselmouth as pm
import numpy as np
from fasttrackpy.processors.smoothers import Smoother
from fasttrackpy.processors import losses
from fasttrackpy.processors import aggs

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from typing import Union

class OneTrack:
    def __init__(
            self,
            sound: pm.Sound,
            maximum_formant: float,
            n_formants: int = 4,
            window_length: float = 0.05,
            time_step: float = 0.002,
            pre_emphasis_from: float = 50,
            smoother:Smoother = Smoother(),
            loss_fun = losses.lmse,
            agg_fun = aggs.agg_sum
        ):
        self.sound = sound
        self.maximum_formant = maximum_formant
        self.n_formants = n_formants
        self.window_length = window_length
        self.time_step = time_step
        self.pre_emphasis_from = pre_emphasis_from
        self.smoother = smoother
        self.loss_fun = loss_fun
        self.agg_fun = agg_fun

        self.formants = self._track_formants()
        self.n_measured_formants = self._get_measured()
        self.imputed_formants = self._impute_formants()
        self.smoothed_formants = self._smooth_formants()
    
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

        return(tracks)

    def _smooth_formants(self):
        smoothed = np.apply_along_axis(
            self.smoother.smooth,
            1,
            self.imputed_formants
        )

        return smoothed
    
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
    def smooth_error(self):
        msqe =  self.loss_fun(
            self.formants, 
            self.smoothed_formants
        )
        error = self.agg_fun(msqe)
        return error