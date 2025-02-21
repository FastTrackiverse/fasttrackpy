from dataclasses import dataclass, field
import numpy as np
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fasttrackpy import OneTrack

@dataclass
class MinMaxHeuristic:
    """
        _summary_

    """
    edge: str = "max"
    measure: str = "frequency"
    number: int = 1
    boundary: float|int|np.floating = 1200

    def eval(self, track: OneTrack):
        """_summary_

        Args:
            track (OneTrack): _description_

        Returns:
            _type_: _description_
        """
        nformants = track.smoothed_formants.shape[0]
        if self.number > nformants:
            return 0
        
        if self.measure == "frequency":
            median_value = np.median(
                track.smoothed_formants[self.number-1,:]
            )
        if self.measure == "bandwidth":
            median_value = np.median(
                track.smoothed_bandwidths[self.number-1,:]
            )

        check = False
        if self.edge == "max":
            check = median_value > self.boundary
        if self.edge == "min":
            check = median_value < self.boundary

        if check:
            return np.inf
        
        return 0


@dataclass
class SpacingHeuristic:
    """_summary_

    """
    top: int = 3
    bottom: int = 0
    diff: float|int|np.floating = 2000
    spacing: float|int|np.floating = 500

    def eval(self, track: OneTrack):
        """_summary_

        Args:
            track (OneTrack): _description_

        Returns:
            _type_: _description_
        """
        nformants = track.smoothed_formants.shape[0]
        if nformants < self.top:
            return 0
        
        top_median = np.median(
            track.smoothed_formants[self.top-1,:]
        )

        if self.bottom == 0:
            bottom_median = 0
        else:
            bottom_median = np.median(
                track.smoothed_formants[self.bottom-1,:]
            )
        
        top_spacing = top_median - bottom_median

        bottom_spacing = np.diff(
            np.median(
                track.smoothed_formants[0:2, :],
                axis = 1
            )
        )

        if top_spacing < self.diff and bottom_spacing < self.spacing:
            return np.inf
        
        return 0