import numpy as np
from pyfusion.acquisition.W7X.get_shot_info import get_shot_utc
from pyfusion.data.convenience import is_listlike

def next_shot(shot, inc=1):
    if is_listlike(shot):
        return((shot[0], shot[1] + inc))
    else:
        return(shot + inc)


def shot_gte(shot1, shot2):
    if is_listlike(shot1):
        if shot1[0] > shot2[0]:
            return(True)
        elif shot1[0] == shot2[0]:
            return(shot1[1] >= shot2[1])
        else:
            return(False)
    else: return(shot1 >= shot2)

def shot_range(shot_from, shot_to, inc=1):
    rng = []
    shot = shot_from
    while True:
        if shot_gte(shot, shot_to):
            return(rng)
        if is_listlike(shot) and get_shot_utc(shot, quiet=True) is None:
            pass # I think this is to avoid dud shots
        else:
            rng.append(shot)
        shot = next_shot(shot, inc=inc)
        # this part is a fudge, but only needed for two component shots
        if is_listlike(shot) and shot[1]>150:   # assume max per day
            shot = (shot[0]+1, 1)

