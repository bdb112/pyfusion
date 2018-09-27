from __future__ import print_function
from six.moves import input
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

def shot_range(shot_from, shot_to=None, inc=1):
    rng = []
    if shot_to is None:
        inp = input(' Assume you want just one shot in a list? (y/nq) ')
        if len(inp) > 0 and inp[0].lower() in 'nq':
            raise ValueError('Need two shots to define a range')
        else: # assume we want to know if the single shot doens't exist
            get_shot_utc(shot_from, quiet=False)
            shot_to = next_shot(shot_from, inc=inc)
            
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

def shot_range_gen(shot_from, shot_to, inc=1):
    shot = shot_from
    while is_listlike(shot) and get_shot_utc(shot, quiet=True) is None:
        shot = next_shot(shot, inc=inc)
    yield shot
    
    
