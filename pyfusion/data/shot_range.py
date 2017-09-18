from pyfusion.acquisition.W7X.get_shot_info import get_shot_utc
import numpy as np


def tupleshot(shot):
    return(isinstance(shot, (tuple, list, np.ndarray)))


def next_shot(shot):
    if tupleshot(shot):
        return((shot[0], shot[1] + 1))
    else:
        return(shot + 1)


def shot_gte(shot1, shot2):
    if tupleshot(shot1):
        if shot1[0] > shot2[0]:
            return(True)
        elif shot1[0] == shot2[0]:
            return(shot1[1] >= shot2[1])
        else:
            return(False)
    else: return(shot1 >= shot2)

def shot_range(shot_from, shot_to):
    rng = []
    shot = shot_from
    while True:
        if shot_gte(shot, shot_to):
            return(rng)
        if tupleshot(shot) and get_shot_utc(shot, quiet=True) is None:
            pass # I think this is to avoid dud shots
        else:
            rng.append(shot)
        shot = next_shot(shot)
        # this part is a fudge, but only needed for two component shots
        if tupleshot(shot) and shot[1]>150:   # assume max per day
            shot = (shot[0]+1, 1)

