from dataclasses import dataclass, field
import numpy as np
from collections.abc import Mapping
from typing import TypeVar, TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from fasttrackpy import OneTrack

TrackType = TypeVar("OneTrack")

@dataclass
class MinMaxHeuristic:
    """
        _summary_

    """
    edge: Literal["min", "max"] = "max"
    measure: Literal["frequency", "bandwidth"] = "frequency"
    number: int = 1
    boundary: float|int|np.floating = 1200

    def eval(self, track: TrackType):
        """_summary_

        Args:
            track (OneTrack): _description_

        Returns:
            _type_: _description_
        """
        nformants = track.n_formants
        if self.number > nformants:
            return 0
        
        if self.measure == "frequency":
            mean_value = np.exp(
                track.log_parameters[self.number-1,0]*
                np.sqrt(2)
            )

        if self.measure == "bandwidth":
            mean_value = np.exp(
                track.bandwidth_parameters[self.number-1, 0]
                *np.sqrt(2)
            )

        check = False
        if self.edge == "max":
            check = mean_value > float(self.boundary)
        if self.edge == "min":
            check = mean_value < float(self.boundary)

        if check:
            return np.inf
        
        return 0


@dataclass
class SpacingHeuristic:
    """_summary_

    """
    top: list[int] = field(default_factory=lambda: [3])
    bottom: list[int] = field(default_factory=lambda: [1,2])
    top_diff: float|int|np.floating = 2000
    bottom_diff: float|int|np.floating = 500

    def __post_init__(self):
        self.top = np.array(self.top)
        self.bottom = np.array(self.bottom)

    def eval(self, track:TrackType):
        """_summary_

        Args:
            track (OneTrack): _description_

        Returns:
            _type_: _description_
        """
        nformants = track.n_formants

        if nformants < self.top.max():
            return 0
        
        top_values = np.array([
            np.exp(track.log_parameters[idx,0]*np.sqrt(2))
            for idx in self.top
        ])

        bottom_values = np.array([
            np.exp(track.log_parameters[idx,0]*np.sqrt(2))
            for idx in self.bottom
        ])

        if top_values.size == 1:
            top_spacing = top_values[0]
        else:
            top_spacing = np.diff(top_values)

        bottom_spacing = np.diff(bottom_values)
    
        if top_spacing < self.top_diff and bottom_spacing < self.bottom_diff:
            return np.inf
        
        return 0
