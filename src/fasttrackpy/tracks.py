import parselmouth as pm
import numpy as np
from fasttrackpy.processors.smoothers import Smoother
from fasttrackpy.processors.losses import Loss
from fasttrackpy.processors.aggs import Agg
from fasttrackpy.processors.outputs import formant_to_dataframe,\
                                           param_to_dataframe,\
                                           get_big_df,\
                                           spectrogram,\
                                           candidate_spectrograms
import matplotlib.pyplot as mp
from aligned_textgrid import SequenceInterval
from aligned_textgrid.sequences.tiers import TierGroup

import polars as pl

from typing import Union
import warnings
import logging


class Track:
    """A generic track class to set up attribute values
    """

    def __init__(
            self,
            sound: pm.Sound,
            n_formants: int = 4,
            window_length: float = 0.05,
            time_step: float = 0.002,
            pre_emphasis_from: float = 50,
            smoother: Smoother = Smoother(),
            loss_fun: Loss = Loss(),
            agg_fun: Agg = Agg()
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
    """_summary_

    Args:
        maximum_formant (float): max formant
        sound (pm.Sound): A `parselmouth.Sound` object.
        n_formants (int, optional): Number of formants to track. Defaults to 4.
        window_length (float, optional): Defaults to 0.05.
        time_step (float, optional):Defaults to 0.002.
        pre_emphasis_from (float, optional): Defaults to 50.
        smoother (Smoother, optional): Smoother function. Defaults to `Smoother` default.
        loss_fun (_type_, optional): Loss function
        agg_fun (_type_, optional): Aggregation function
    
    Attributes:
        maximum_formant (float): The max formant
        time_domain (np.array): The time domain of the formant estimates
        formants (np.ndarray): A (formants, time) array of values. The formants
            as initially estimated by praat-parselmouth
        n_measured_formants (int): The total number of formants for which
            formant tracks were estimatable
        smoothed_formants (np.ndarray): The smoothed formant values, using 
            the method passed to `smoother`.
        smooth_error (float): The error term between formants and 
            smoothed formants.
    """

    def __init__(
            self,
            maximum_formant: float,
            sound: pm.Sound,
            n_formants: int = 4,
            window_length: float = 0.025,
            time_step: float = 0.002,
            pre_emphasis_from: float = 50,
            smoother: Smoother = Smoother(),
            loss_fun: Loss = Loss(),
            agg_fun: Agg = Agg()
        ):
        super().__init__(
            sound=sound,
            n_formants=n_formants,
            window_length=window_length,
            time_step=time_step,
            pre_emphasis_from=pre_emphasis_from,
            smoother=smoother,
            loss_fun=loss_fun,
            agg_fun=agg_fun
        )
        self.maximum_formant = maximum_formant

        self.formants, self.time_domain = self._track_formants()
        self.n_measured_formants = self._get_measured()
        self.smoothed_list = self._smooth_formants()
        self._file_name = None
        self._id = None      
        self._group = None  
        self._formant_df = None
        self._param_df = None
        self._interval = None
    
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
            for x in self.formants
        ]
        return smoothed_list
    
    def _get_measured(self):
        nan_tracks = np.isnan(self.formants)
        mostly_nan = np.mean(nan_tracks, axis = 1) > 0.5
        if np.any(mostly_nan):
            return np.argmax(mostly_nan)
        return mostly_nan.shape[0]
    
    @property
    def smoothed_formants(self):
        return np.array(
            [x.smoothed for x in self.smoothed_list]
        )

    @property
    def parameters(self):
        return np.array(
            [x.params for x in self.smoothed_list]
        )

    @property
    def smooth_error(self):
        msqe =  self.loss_fun.calculate_loss(
            self.formants[0:self.n_measured_formants], 
            self.smoothed_formants[0:self.n_measured_formants]
        )
        error = self.agg_fun.aggregate(msqe)
        return error
    
    @property
    def file_name(self):
        return self._file_name
    
    @file_name.setter
    def file_name(self, x):
        self._file_name = x
    
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, x):
        self._id = x

    @property
    def interval(self):
        return self._interval
    
    @property
    def group(self):
        return self._group
    
    @group.setter
    def group(self, groupname):
        self._group = groupname

    @interval.setter
    def interval(self, interval: SequenceInterval):
        self._interval = interval
        self.label = interval.label
        self.id = interval.id
        self.group = self.__get_group(self._interval)

    def __get_group(self, interval):
        if isinstance(interval.within, TierGroup):
            return interval.within.name
        
        return self.__get_group(interval.within)

    def to_df(self, output = "formants"):
        if output == "formants"\
              and not isinstance(self._formant_df, pl.DataFrame):
            df =  formant_to_dataframe(self)
            self._formant_df = df
            return df
        if output == "formants":
            return self._formant_df
        if output == "param"\
            and not isinstance(self._param_df, pl.DataFrame):
            df =  param_to_dataframe(self)
            self._param_df = df
            return df
        if output == "param":
            return self._param_df
        
        raise ValueError("output must be either 'formants' or 'param'")
    
    def spectrogram(self, **kwargs):
        spectrogram(self, **kwargs)
    
        

class CandidateTracks(Track):
    """A class for candidate tracks for a single formant
    
    This takes the same arguments as `OneTrack` except for `max_formant.

    Args:
        min_max_formant (float, optional): The floor for the `max_formant` setting. 
            Defaults to 4000.
        max_max_formant (float, optional): The cieling for the `max_formant` setting.
            Defaults to 7000.
        nstep (int, optional): The number of steps for the grid search.
            Defaults to 20.

    Attributes:
        candidates (list[OneTrack,...]): A list of `OneTrack` tracks.
        min_n_measured (int): The smallest number of successfully measured 
            formants across all `candidates`
        smooth_errors (np.array): The error terms for each treack in `candidates`
        winner_idx (int): The candidate track with the smallest error term
        winner (OneTrack): The winning `OneTrack` track,
    """

    def __init__(
        self,
        sound: pm.Sound,
        min_max_formant: float = 4000,
        max_max_formant: float = 7000,
        nstep = 20,
        n_formants: int = 4,
        window_length: float = 0.025,
        time_step: float = 0.002,
        pre_emphasis_from: float = 50,
        smoother: Smoother = Smoother(),
        loss_fun: Loss = Loss(),
        agg_fun: Agg = Agg()
    ):
        super().__init__(
            sound=sound,
            n_formants=n_formants,
            window_length=window_length,
            time_step=time_step,
            pre_emphasis_from=pre_emphasis_from,
            smoother=smoother,
            loss_fun=loss_fun,
            agg_fun=agg_fun
        )

        self.min_max_formant = min_max_formant
        self.max_max_formant = max_max_formant
        self.nstep = nstep
        self.max_formants = np.linspace(
            start = self.min_max_formant,
            stop = self.max_max_formant,
            num = self.nstep
        )
        self._file_name = None
        self._id = None
        self._formant_df = None
        self._param_df = None
        self._interval = None

        self.candidates = [
            OneTrack(
                sound = self.sound,
                maximum_formant=x,
                n_formants = self.n_formants,
                window_length = self.window_length,
                time_step = self.time_step,
                pre_emphasis_from = self.pre_emphasis_from,
                smoother = self.smoother,
                loss_fun = self.loss_fun,
                agg_fun = self.agg_fun
                
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

    @property
    def file_name(self):
        return self._file_name
    
    @file_name.setter
    def file_name(self, x):
        self._file_name = x
        for c in self.candidates:
            c.file_name = x
    
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, x):
        self._id = x
        for c in self.candidates:
            c.id = x

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, interval):
        self._interval = interval
        for c in self.candidates:
            c.interval = interval

    def _normalize_n_measured(self):
        for track in self.candidates:
            track.n_measured_formants = self.min_n_measured
    
    def to_df(self, which = "winner", output = "formants"):
        if which == "winner":
            return self.winner.to_df(output=output)
        
        if output == "formants"\
            and not isinstance(self._formant_df, pl.DataFrame):
            big_df = get_big_df(self, output=output)
            self._formant_df = big_df
            return big_df
        
        if output == "formants":
            return self._formant_df
        
        if output == "param"\
            and not isinstance(self._param_df, pl.DataFrame):
            big_df = get_big_df(self, output=output)
            self._param_df = big_df
            return big_df
        
        if output == "param":
            return self._param_df
            
    def spectrograms(self, **kwargs):
        candidate_spectrograms(self, **kwargs)

