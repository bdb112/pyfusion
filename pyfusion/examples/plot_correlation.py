""" plot correlation of two diagnostics over a time range

for cpt in 'Axial,Outward,Upward'.split(','):runpy("pyfusion/examples/plot_correlation.py diag1='H1Toroidal{}' diag2='ElectronDensity_1' bp=[20e3,2e3] shot_number=100618 hold=1".format(cpt))

# good test of axis labelling (2D)
run pyfusion/examples/plot_correlation dev_name='W7X' diag1='W7X_L57_LPALLI' diag2='W7X_L57_LP01_04' shot_number=[20160308,12] t_range=[1.291,1.298] bp=[50e3,98e3]

# good test of axis labelling (1D)
run pyfusion/examples/plot_correlation dev_name='W7X' diag1='W7X_L57_LPALLI' diag2='W7X_L57_LP20_I' shot_number=[20160308,12] t_range=[1.291,1.298] bp=[50e3,98e3]

# this needs to be run several times
 [pyfusion.GL.remove(d) for d in pyfusion.GL if '[20e3,18e3]' in d['vardict']['bp']]

# or to just plot those not containing this
lst=unique([d['vardict']['bp'] for d in pyfusion.GL]).tolist(); lst.remove('[20e3,18e3]')
for sel in  lst: plot(*(transpose([[d['shot']-100000, d['diag2']] for d in pyfusion.GL if sel in d['vardict']['bp'] and 'cmd' in d and d['cmd']==pyfusion.GL[-1]['cmd']])),label=sel)
_PYFUSION_TEST_@@dev_name=W7X diag1=W7X_L57_LP07_I diag2=W7X_L57_LP08_I shot_number=[20160308,12] t_range=[1.285,1.298]
diag2 = 'H1ToroidalMirnov_15y'

"""
import pyfusion
import sys
import matplotlib.pyplot as plt
import numpy as np
from pyfusion.data.pyfusion_corrinterp import correlation
import time as tm  # for GL stamping

_var_defaults = """
dev_name = "H1Local"
diag1 = 'H1ToroidalAxial'
diag2 = 'H1ToroidalMirnov_15y'
shot_number = 100617
# shot_number = [20160308,39]
# dev_name = "LHD"; diag_name = "MP1" ; shot_number = 27233
t_range = []
plotit=1  # can tun plots off when run in loops by runpy for example
vsfreq = dict(plot=1)
hold = 0
mask1 = None  # [0,-1] wipe out the wonky channels
mask2 = None
labfmt='{diag1} x {diag2}: {shot_number}'
coefft = 1
bp = [] # tuple [freq (Hz), bw(Hz - if absent, 0.2*centre), order (not implemented)]  - if longer, use as a digital filter
mindf = 500  # minimum difference in pass and stop freqgs
debug=None
"""
exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data1 = dev.acq.getdata(shot_number, diag1)
data2 = dev.acq.getdata(shot_number, diag2)

if len(t_range) > 0:
    d1 = data1.reduce_time(t_range)
    d2 = data2.reduce_time(t_range)
else:  # use d1 so that original data is available for interative use
    d1 = data1
    d2 = data2
    
if len(bp) > 0:
    bw = bp[1] if len(bp) > 1 else 0.1 * bp[0]
    passband = [bp[0] - bw/2., bp[0] + bw/2.]
    stopband = [passband[0] - min(mindf, bw/2), passband[1] + min(mindf,(bw/2))]
    d1 = d1.filter_fourier_bandpass(passband=passband, stopband=stopband, debug=debug)
    d2 = d2.filter_fourier_bandpass(passband=passband, stopband=stopband)

if list(vsfreq) !=[]:
    if len(d1.keys()) > 1 or len(d2.keys()) > 1:
        raise ValueError('freq plot only makes sense for single channels')
    
    plotit = 1  # force plots
    [[f, Cxy]] = correlation(d1, d2, vsfreq=vsfreq)
    plt.loglog(f, Cxy)
    plt.show(0)
    sys.exit()
    
if plotit:
    if hold==0: fig, ax = plt.subplots(1,1)
    else:
        fig = plt.gcf()
        ax = plt.gca()
corr = np.array(correlation(d1, d2, coefft=coefft))

if mask1 is not None:
    corr[mask1,:,:] = np.nan

if mask2 is not None:
    corr[:,mask2,:] = np.nan

if plotit:
    if len(np.shape(corr)) == 3:
        CS = ax.imshow(np.array(corr)[:,:,0],interpolation='nearest')
        plt.colorbar(CS)
    else:
        ax.step(range(len(corr)), corr, label=labfmt.format(**locals()), where='mid')

    plt.legend(fontsize='small')
    titl = str('shot {sh}, {c}'.format(sh=shot_number, c=['(phys_units)','coefft'][coefft]))
    if len(bp)>0:
        titl += ', bandpass({f:.1f}kHz +/-{bw2:.0f}%) '.format(f=bp[0]/1e3, bw2=100*bw/bp[0]/2.)
    ax.plot(ax.get_xlim(), [0,0], 'k', lw=0.05, )
labels1 = data1.keys()
labels2 = data2.keys()
if len(labels1) == 1:
       labels1, labels2 = labels2, labels1

if len(labels1) != np.shape(corr)[-2]:
    print('label mismatch?', labels1)
else:
    if plotit:
        ax.set_xticks(range(len(labels1)))
        ax.set_xticklabels(labels1, rotation=90, fontsize=['','x-','xx-'][int(np.log(len(labels1[0]))/1.3)] + 'small')
        if len(np.shape(corr)) == 3:
            ax.set_yticks(range(len(labels2)))
            ax.set_yticklabels(labels2, rotation=0, fontsize=['','x-','xx-'][int(np.log(len(labels2[0]))/1.3)] + 'small')
        plt.title(titl)
        plt.show(0)

corr_nonan = np.array(corr)[np.where(~np.isnan(corr))]  
RMSCorr = np.sqrt(np.mean(np.power(corr_nonan,2)))
print(shot_number, RMSCorr)
GL.append(dict(diag2=RMSCorr, shot=shot_number, time=tm.time()))
# GD.update({shot_number: RMSCorr})  # originally I used a dict

"""
ft = [(correlation(d1.reduce_time([t,t+dt], copy=1), d2.reduce_time([t,t+dt],copy=True), vsfreq=dict(nperseg=256)))[0][1] for t in linspace(min(d1.timebase), -0.001+max(d1.timebase), 10, endpoint=0)]
imshow ft[0], aspect='auto'
imshow ft, aspect='auto'

"""
