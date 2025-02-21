from dataclasses import dataclass, field
import numpy as np
from collections.abc import Mapping
from typing import TypeVar, TYPE_CHECKING, Literal, Annotated
if TYPE_CHECKING:
    from fasttrackpy import OneTrack

TrackType = TypeVar("OneTrack")

@dataclass
class MinMaxHeuristic:
    """
    A heuristic class for min/max frequencies and bandwidths

    Examples:
        To define an F1 maximum:

        ```python
        F1_Max = MinMaxHeuristic(    
            edge="max",
            measure="frequency",
            number=1,
            boundary=1200
            )
        ```

    Args:
        edge (Literal["min", "max"]):
            Whether this heuristic defines a min or a max
        measure: (Literal["frequency", "bandwidth"]):
            Whether this heuristic is defined over frequencies 
            or bandwidths
        number (int):
            The formant number
        boundary (float|int|np.floating):
            The min or max value.
    """
    edge: Literal["min", "max"] = "max"
    measure: Literal["frequency", "bandwidth"] = "frequency"
    number: int = 1
    boundary: float|int|np.floating = 1200

    def eval(self, track: TrackType):
        """
        Evaluate whether or not the track passes the 
        heuristic

        Args:
            track (OneTrack):
                The track to evaluate

        Returns:
            (Literal[0|np.inf]):
                If it passes the heuristic, 0. 
                If it doesn't, `np.inf`
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
    """
    A class for defining formant spacing heuristics.

    Args:
        top (list[int]): 
            The top formants to evaluate spacing on
            (or value, if the list is only 1 value long).
        bottom (list[int]):
            The bottom formants to evaluate spacing on.
        top_diff (float|int|np.floating):
            The spacing of the top formants
        bottom_diff (float|int|np.floating):
            The spacing of the bottom formants
    """
    top: list[int] = field(default_factory=lambda: [3])
    bottom: list[int] = field(default_factory=lambda: [1,2])
    top_diff: float|int|np.floating = 2000
    bottom_diff: float|int|np.floating = 500

    def __post_init__(self):
        self.top = np.array(self.top)
        self.bottom = np.array(self.bottom)

    def eval(self, track:TrackType):
        """
        Evaluate whether or not the track passes
        the heuriustic

        Args:
            track (OneTrack):
                The track to evaluate.

        Returns:
            (Literal[0|np.inf]):
                If the track passes, 0.
                If the track doesn't pass, `np.inf`.
        """
        nformants = track.n_formants

        if nformants < self.top.max():
            return 0
        
        top_values = np.array([
            np.exp(track.log_parameters[idx-1,0]*np.sqrt(2))
            for idx in self.top
        ])

        bottom_values = np.array([
            np.exp(track.log_parameters[idx-1,0]*np.sqrt(2))
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

F1_Max: Annotated[
    MinMaxHeuristic, 
    "F1 should not be greater than 1200 hz"
] = MinMaxHeuristic(
    edge="max",
    measure="frequency",
    number=1,
    boundary=1200
)
"""
[](`~fasttrackpy.processors.heuristic.MinMaxHeuristic`): F1 should not be greater than 1200 hz
"""

B2_Max: Annotated[
    MinMaxHeuristic, 
    "B2 should not be greater than 500 hz"
] = MinMaxHeuristic(
    edge="max",
    measure="bandwidth",
    number=2,
    boundary=500
)
"""
[](`~fasttrackpy.processors.heuristic.MinMaxHeuristic`): B2 should not be greater than 500 hz
"""

B3_Max: Annotated[
    MinMaxHeuristic, 
    "B3 should not be greater than 600 hz"
] = MinMaxHeuristic(
    edge="max",
    measure="bandwidth",
    number=3,
    boundary=600
)
"""
[](`~fasttrackpy.processors.heuristic.MinMaxHeuristic`): B3 should not be greater than 600 hz
"""

F4_Min:  Annotated[
    MinMaxHeuristic, 
    "F4 should not be less than 2900 Hz"
] = MinMaxHeuristic(
    edge="min",
    measure="frequency",
    number=4,
    boundary=2900
)
"""
[](`~fasttrackpy.processors.heuristic.MinMaxHeuristic`): F4 should not be less than 2900 Hz
"""

Rhotic: Annotated[
    SpacingHeuristic, 
    "If F3 < 2000 Hz, F1 and F2 should be at least 500 Hz apart."
] = SpacingHeuristic(
    top=[3],
    bottom=[1,2],
    top_diff=2000,
    bottom_diff=400
)
"""
[](`~fasttrackpy.processors.heuristic.SpacingHeuristic`): If F3 < 2000 Hz, F1 and F2 should be at least 500 Hz apart.
"""

F3_F4_Sep: Annotated[
    SpacingHeuristic, 
    "If F4 - F3 < 500 Hz, F2-F1 > 1500."
] = SpacingHeuristic(
    top=[3,4],
    bottom=[1,2],
    top_diff=500,
    bottom_diff=1500
)
"""
[](`~fasttrackpy.processors.heuristic.SpacingHeuristic`): If F4 - F3 < 500 Hz, F2-F1 > 1500.
"""