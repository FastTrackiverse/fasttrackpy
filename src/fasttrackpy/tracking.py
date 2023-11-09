import parselmouth as pm
import numpy as np
import scipy.fft
from fasttrackpy import smoothers
from fasttrackpy import losses
from fasttrackpy import aggs

def all_tracks(sound, 
               start = 5000, 
               stop = 7000, 
               nstep = 20,
               time_step = 0.001,
               n_formants = 6,
               window_length = 0.025,
               pre_emphasis_from = 50):
    """
    Extract all formant tracks

    Args:
        sound (parselmouth.Sound): Sound object read in by `parselmouth`
        start (float): lowest max-formant
        stop (float): highest max-formant
        nstep (int): number of steps along max-formant
        time_step (float): time step of formant tracks (in sec)
        n_formants (float): Number of formants (2*`n_formants`) must be an integer
        window_length (float): analysis window size
        pre_emphasis_from (float): pre-emphasis
    
    Returns:
        (numpy.ndarray): array of all formant tracks with shape (n_formants, :, nstep)
    """
    max_formants = np.linspace(start = start, stop = stop, num = nstep)
    formant_arrays = [
        findformants(maximum_formant = x,
                     sound=sound,
                     time_step = time_step,
                     n_formants = n_formants, 
                     window_length = window_length,
                     pre_emphasis_from = pre_emphasis_from)
        for x in max_formants
    ]
    formant_candidates = np.stack(formant_arrays, axis = -1)
    return(formant_candidates)

def choose_winner(candidates, 
                  smooth_fun = smoothers.dct_smooth,
                  loss_fun = losses.lmse,
                  agg_fun = aggs.agg_sum,
                  args = {"smooth_kwargs": {},
                          "loss_kwargs": {},
                          "agg_kwargs": {} }):
    """
    Return index of winner
    """

    smoothed = smooth_formants(candidates, smooth_fun, **args["smooth_kwargs"])
    all_mses = smooth_error(candidates, smoothed, loss_fun, **args["loss_kwargs"])
    agg_mses = agg_fun(all_mses, **args["agg_kwargs"])
    winner_idx = np.nanargmin(agg_mses)
    return(winner_idx)
    
def findformants(maximum_formant,
                sound, 
                n_formants=4,  
                window_length = 0.05, 
                time_step = 0.002, 
                pre_emphasis_from = 50):
    """
    Track formants
    """
    formants = sound.to_formant_burg(
        time_step = time_step,
        max_number_of_formants = n_formants, 
        maximum_formant = maximum_formant,
        window_length = window_length,
        pre_emphasis_from = pre_emphasis_from
    )

    time_domain = formants.xs()
    tracks = np.array(
        [
            [formants.get_value_at_time(i+1, x) 
                for x in time_domain] 
            for i in range(int(np.floor(n_formants)))
        ]
    )
    return(tracks)

def smooth_error(formants, smoothed, 
                 loss_fun = losses.lmse, 
                 **kwargs):
    """
    calculate error
    """

    loss = loss_fun(formants, smoothed, **kwargs)
    return(loss)

def smooth_formants(formants, 
                    smooth_fun = smoothers.dct_smooth, 
                    axis = 1, 
                    **kwargs):
    """
    Given a formants array, smooth it according to smooth_fun
    """
    smoothed = np.apply_along_axis(smooth_fun, axis, formants, **kwargs)
    return(smoothed)

if __name__ == "__main__":
    pass