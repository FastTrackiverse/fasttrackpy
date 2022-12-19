import parselmouth as pm
import numpy as np

def findformants(sound, 
                 n_formants=4,
                 maximum_formant = 7000, 
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
            for i in range(n_formants)
        ]
    )
    return(tracks)
