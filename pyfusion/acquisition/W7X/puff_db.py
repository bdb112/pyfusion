""" a crude 'database' of off-the-books puffing - i.e. not in the data system

"""
import numpy as np
import matplotlib.pyplot as plt
import pyfusion
from scipy.interpolate import griddata

puff_db = {}
delta = 1e-6

def puffadd(self, key, **kwargs):
    self.update({key: dict(**kwargs)})
# Data from Oliver Schmit's commented shot list, for N2, Ar, igore the estimated density - they
# seemed to be leaving it alone at 2e19

puffadd(puff_db, (20160309, 42), times=[100, 150, 160, 190, 210, 230, 260, 280, 320, 340], nd=300, gas='N2')
puffadd(puff_db, (20160309, 43), times=[100,300], nd=200, gas='N2')
puffadd(puff_db, (20160309, 44), times=[100,500], nd=300, gas='N2')
puffadd(puff_db, (20160309, 50), times=[100, 300], nd=100, gas='N2')
# HM51 100 
puffadd(puff_db, (20160309, 51), times=[100, 300], nd=100, gas='N2')
# actually two gas boxes at 100 each
puffadd(puff_db, (20160309, 52), times=[100, 300], nd=200, gas='N2')

puffadd(puff_db, (20160310, 40), times=[100, 175, 250, 325, 400, 475, 550, 625, 700, 775 ], nd=145, gas='Ar')


def get_puff(shot, t_range=None, numpoints=500, debug=0):
    shot = tuple(shot)
    if shot not in puff_db:
        return(None)
    dct = puff_db[shot]
    times = (np.array(dct['times'])/1000.0).tolist()  # to seconds
    nd18 = np.array(dct['nd'])/100.
    divis = '/100mBar'
    if len(np.shape(nd18)) == 0:
        nd18 = nd18 * np.ones(len(times))
    if t_range is None:
        t_range = [0,1.1*np.max(times)]
    times.insert(0, t_range[0])
    times.append(t_range[1])
    tm = np.linspace(t_range[0], t_range[1], numpoints) 
    puff_t = []
    puff_n = []
    for (i, pt) in enumerate(times):
        if (i == 0) or (i == len(times)-1):
            puff_t.append(pt)
            puff_n.append(0)
        else: 
            puff_t.append(pt-delta)
            puff_n.append(puff_n[-1])  # previous
            puff_t.append(pt+delta)
            puff_n.append(nd18[i] if puff_n[-1] == 0 else 0)  # opposite
    if debug > 0: plt.plot(puff_t, puff_n,'o')
    grid_nd18 = griddata(np.array(puff_t), np.array(puff_n), (tm), method='linear')
    if debug > 0: plt.plot(tm, grid_nd18)
    return(tm, grid_nd18, dct['gas']+divis)
