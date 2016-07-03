import numpy as np
import matplotlib.pyplot as plt
import pyfusion
from pyfusion.data.DA_datamining import Masked_DA, DA

from lukas.PickledAssistant import lookupPosition

#from cycler import cycler  # not working yet

def rot(x, y, t):
    from numpy import cos, sin
    return(x*cos(t)+y*sin(t), -x*sin(t)+y*cos(t))

def LCFS_plot(da, diag, t0_utc, t_range, ax=None, labelpoints=0):#False):
    ax = plt.gca() if ax is None else ax
    chans = np.array(da['info']['channels'])
    dt = (t0_utc - da['info']['params']['i_diag_utc'][0])/1e9
    indrange = np.searchsorted(da['t_mid'], np.array(t_range) + dt)
    inds = range(*indrange)
    if len(inds) < 3:
        ll = len(inds)
        pyfusion.logging.warn('{msg} samples ({ll}) in the time range {t}'
                              .format(msg=['!No! ','too few'][ll>0], ll=ll, t=t_range))
    sig = np.nanmean(da.masked[diag][inds], axis=0)
    if 'e'+diag in da:
        ebars = np.nanmean(da['e'+diag][inds], axis=0)
    else:
        ebars = None

    distLCFS = []
    y = [] # perp dist of probe from limiter midplane
    exclude = []
    if diag in  ['ne18']:
        exclude.extend(['LP11','LP12','L57_LP09'])
    for (c, ch) in enumerate(chans):
        if sum([st in ch for st in exclude]):
            sig[c] = np.nan
        LPnum = int(ch[-2:])
        lim = 'lower' if 'L57' in ch else 'upper'
        X, Y, Z, dLCFS = lookupPosition(LPnum, lim)
        y.append(rot(X, Y,  2*np.pi*4/5.)[1])
        distLCFS.append(dLCFS)
        #print()
        if labelpoints and (not np.isnan(sig[c])):
            ax.text(np.sign(y[c])*distLCFS[c], sig[c], ch)
    ax.errorbar(np.sign(y)*distLCFS, sig, ebars, fmt='o', label=lim)
    ax.set_ylabel(diag)
    ax.set_ylim(0, 1.2*np.nanmax(sig))


def infostamp(txt, fig=None, default_kwargs=dict(horizontalalignment='right', verticalalignment='bottom', fontsize=6, x=0.99, y=0.008), **kwargs):
    import os, sys, pyfusion

    fig = plt.gcf() if fig is None else fig
    actual_kwargs = default_kwargs.copy()
    actual_kwargs.update(kwargs)
    extra = ' '.join(['pyfusion V ', pyfusion.VERSION,os.getlogin(),os.getcwd(), sys.version[:12]])
    plt.figtext(s=txt+extra, figure=fig, **actual_kwargs)

dev_name = 'W7X'
da53 = DA('LP/LP20160309_10_L53_2k2.npz')
da57 = DA('LP/LP20160309_10_L57_2k2.npz')
t_range = [0.91, 0.93]
t_range = [0.51, 0.53]
"""
# t_range = [0.61, 0.63]
# t_range = [1.15, 1.18]
da53 = DA('LP/LP20160309_52_L53_2k2.npz')
da57 = DA('LP/LP20160309_52_L57_2k2.npz')
t_range = [0.5, 0.52]
t_range = [0.45, 0.47]
t_range = [0.48, 0.5]
#t_range = [0.3, 0.32]
"""

shot_number = [da53['date'][0], da53['progId'][0]]
dev = pyfusion.getDevice(dev_name)
echdata = dev.acq.getdata(shot_number,'W7X_TotECH')
wech = np.where(echdata.signal > 100)[0]
tech = echdata.timebase[wech[0]]
t0_utc = int(tech * 1e9) + echdata.utc[0]


figLCFS, (axLCTe, axLCne) = plt.subplots(2, 1)
for ax in (axLCTe, axLCne):
    ax.set_color_cycle(['b', 'r', 'y', 'g', 'orange', 'c', 'm'])

for da in [da53, da57]:
    LCFS_plot(da, 'ne18', t0_utc=t0_utc, t_range=t_range, ax=axLCne)
    LCFS_plot(da, 'Te', t0_utc=t0_utc, t_range=t_range, ax=axLCTe)

for ax in (axLCTe, axLCne):
    ax.legend(loc='best')
    ax.set_xlabel('distance to LCFS (from Lukas R: -ve for left side)')
    #ax.set_prop_cycle(cycler('color', ['b', 'r', 'y', 'k']))

figLCFS.suptitle('shot {s}, time {fr}-{t}s into ECH'
                 .format(s=[da['date'][0], da['progId'][0]],fr=t_range[0],t=t_range[1]))
infostamp(' '.join([da53.name, da57.name]))
figLCFS.show()
