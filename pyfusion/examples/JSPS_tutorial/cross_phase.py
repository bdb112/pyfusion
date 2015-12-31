"""
Plot coherence and cross phase for a multi signal diagnostic
Can pass a pyfusion data item or get data from a source
H-1 88673 is a nice example

This version reproduces figure 2 in the tutorial article. (see tut_notes.md)
see pyfusion/examples for a simpler version.  When run from the download 
package, a downsmapled data file is used, so we need NFFT=256 to obtain thetasame time window

mlab.cohere_pairs is > 10x faster on 15 signals than separate cohere() calls
See also csd in  /usr/lib/pymodules/python2.7/matplotlib/pyplot.py

"""

import pyfusion as pf
import numpy as np       
import matplotlib.pyplot as plt
import os
from numpy import mod, pi

_var_defaults = """
dev_name = "H1Local"           # "LHD"
diag_name = "H1ToroidalAxial"  # "MP1"
shot_number = 88673            # 27233
NFFT=256                       # 2048 for full data, 256 for downsampled
noverlap=1.0
unit_freq = 1000
dat=None
time_range=[0.04,0.05]
omit=[]
lw=1
lwtot=2*lw
chans=None
nr=2
shade=[22.5,24.5]   # None supresses the shaded area
"""
exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

pf.config.set('global','localdatapath',os.path.join(pf.PYFUSION_ROOT_DIR,'examples','JSPS_tutorial','local_data')) # this line points to local data


if nr > 0:
    try:  # use my nicer version if it is available
        from subplots import subplots
        spkwargs=dict(apportion=[2,1])
    except ImportError:
        from matplotlib.pyplot import subplots
        spkwargs={}

    fig, axs = subplots(nrows=nr,ncols=1, sharex='all',**spkwargs)

if nr==1:
    strtpair = 0
    ax1, ax2 = axs, axs
    plt.sca(ax2)
else:
    strtpair = 1
    ax1, ax2 = axs
    plt.sca(axs[0])

xlab = {1:'Hz', 1000:'kHz', 1000000:'MHz'}[unit_freq] 

if type(noverlap) != int:
    noverlap = int(noverlap*NFFT)

if dat is None:
    dev = pf.getDevice(dev_name)
    orig_data = dev.acq.getdata(shot_number,diag_name)
    data = orig_data
else:
    data = dat

if time_range[1] != time_range[0]:
    data = data.reduce_time(time_range)

dt = np.average(np.diff(data.timebase))

#%run pyfusion/examples/plot_signals shot_number=88672 diag_name='H1ToroidalAxial' dev_name='H1Local'
X = np.array([s for s in data.signal]).T

if chans is None: 
    chans = [i for i in range(np.shape(X)[1]-1)]
for o in omit:
    chans.remove(o)

pairs = [(chans[i],chans[i+1]) for i in range(len(chans)-1)]

pairs.insert(0, (pairs[0][0],pairs[-1][1]))  #  the first elt is from end to end
coh,cp,freq = plt.mlab.cohere_pairs(X,pairs,NFFT=NFFT,Fs=1/dt/float(unit_freq))

colorset=('b,g,r,c,m,y,k,orange,purple,lightgreen,gray,yellow,brown,teal,tan'.split(',')) # to be rotated

for (i,pair) in enumerate(pairs[strtpair:]): 
    plt.plot(freq,coh[pair],label='dph'+str(pair),color=colorset[i],linewidth=lw)
    plt.plot(freq,cp[pair],color=colorset[i],linestyle=['--','-'][i==0 and strtpair==0],
            linewidth=lw)
    
plt.setp(ax2, ylim=(-4,4), ylabel = 'phase difference (rad) || coherence')
totph=np.sum([cp[pair] for pair in pairs[0:-1]],0)
hist = data.history
sz = 'small' if len(hist.split('\n'))>3 else 'medium'
plt.title(hist + str(': NFFT={NFFT}, noverlap={noverlap}'
                            .format(NFFT=NFFT, noverlap=noverlap)),size=sz)
if nr>1:
    plt.ylim(-2.5,1.2)
    plt.legend(prop={'size':'small'},loc='lower right')

if shade is not None:  # shade the area of interest for the tutorial
    from matplotlib.patches import Rectangle
    someX, someY = 23.5, -0.5
    ax1.add_patch(Rectangle((shade[0], someY - 2), shade[1]-shade[0], 4, facecolor="lightcyan",
                            edgecolor='lightgray'))

ax2.plot (freq,totph,':',label='total phase',color='b',linewidth=lwtot)
if nr>1: plt.setp(ax2, ylim=(-14.5,-12.5), ylabel='phase (rad)')
plt.setp(ax2.get_yticklabels()[-1],visible=False)
# plot a few to make sure at lest one is in range
for offs in [0, 2]:
    ax2.plot(freq, mod(totph,2*pi)-offs*pi,':',color='b',linewidth=lwtot)
endendpair = pairs[0]
for offs in [0,1,2]:
    ax2.plot(freq,cp[endendpair]-offs*2*pi,color='r',
             linewidth=lwtot,label=['dph'+str(endendpair),'',''][offs])
ax2.legend(prop={'size':'small'},loc='lower right')
xl = [15,30] if nr>1 else [1,100] # lower limit of 1 is better for log x scale
plt.setp(ax2, xlabel=str('frequency ({xlab})'.format(xlab=xlab)), xlim = xl)   


plt.show()

# title('')
