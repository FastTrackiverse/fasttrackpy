import parselmouth as pm
import numpy as np
from fasttrackpy.processors import smoothers
from fasttrackpy.processors import losses
from fasttrackpy.processors import aggs

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
            smoother = smoothers.dct_smooth
        ):
        self.sound = sound
        self.maximum_formant = maximum_formant
        self.n_formants = n_formants
        self.window_length = window_length
        self.time_step = time_step
        self.pre_emphasis_from = pre_emphasis_from
        self.smoother = smoother
    def __repr__(self):
        return f"A formant track object. {self.formants.shape}"

    @property
    def formants(self):
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


