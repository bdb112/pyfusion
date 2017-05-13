"""
Plot coherence and cross phase for a multi signal diagnostic
Can pass a pyfusion data item or get data from a source
H-1 88673 is a nice example

mlab.cohere_pairs is > 10x faster on 15 signals than separate cohere() calls
See also csd in  /usr/lib/pymodules/python2.7/matplotlib/pyplot.py
The fully annotated version of this is in examples/JSPF_tutorial
This version now has some of the features of the above, and extends those (e.g.nr=3)  also bugs fixed - e.g.
Did not allow correctly for the 0-14 phase to be put at the top of the list (so sum was wrong).  
"""

import pyfusion
import matplotlib.pyplot as plt
import numpy as np
from numpy import mod, pi

_var_defaults = """
dev_name = "H1Local"           # "LHD"
diag_name = "H1ToroidalAxial"  # "MP1"
shot_number = 88673            # 27233
NFFT=2048                      # 1024
noverlap=1.0
unit_freq = 1000
dat=None
time_range=[0,0]
omit=[]
lw=1
lwtot=2*lw
chans=None
nr=2
shade=[22.5,24.5]   # None supresses the shaded area
mono=False          # JSPF wants mono pdfs
strtpair=0
"""
exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if nr > 0:
    try:  # use my nicer version if it is available
        from subplots import subplots
        # close up space between graphs
        spkwargs=dict(apportion=[[1],[1],[2,1],[1]][nr],spadj_kwargs=dict(hspace=0))
    except ImportError:
        from matplotlib.pyplot import subplots
        spkwargs={}

    fig, axs = subplots(nrows=nr,ncols=1, sharex='all',**spkwargs)

if nr==1:
    strtpair = 0
    ax1, ax2, ax3 = axs, axs, axs
    plt.sca(ax2)
elif nr == 2:
    strtpair = 1
    ax1, ax3 = axs
    ax2 = ax1
    plt.sca(axs[0])
elif nr == 3:
    ax1, ax2, ax3 = axs
else: 
    raise ValueError('nr should be in [1,2,3]')

xlab = {1:'Hz', 1000:'kHz', 1000000:'MHz'}[unit_freq] 

if type(noverlap) != int:
    noverlap = int(noverlap*NFFT)

if dat is None:
    dev = pyfusion.getDevice(dev_name)
    orig_data = dev.acq.getdata(shot_number,diag_name)
    data = orig_data
else:
    data = dat

if time_range[1] != time_range[0]:
    data = data.reduce_time(time_range)

dt = np.average(np.diff(data.timebase))

X = np.array([s for s in data.signal]).T

if chans is None: 
    chans = [i for i in range(np.shape(X)[1])]
for o in omit:
    chans.remove(o)

pairs = [(chans[i],chans[i+1]) for i in range(len(chans) - 1)]

pairs.insert(0, (pairs[0][0],pairs[-1][1]))  #  the first elt is from end to end
if np.shape(X)[0]//2 < NFFT:
    raise ValueError('need more datapoints for given NFFT')

coh,cp,freq = plt.mlab.cohere_pairs(X,pairs,NFFT=NFFT,Fs=1/dt/float(unit_freq))

colorset=('b,g,r,c,m,y,k,orange,purple,lightgreen,gray,yellow,brown,teal,tan'.split(',')) # to be rotated

#ax = plt.gca()
# go through the pairs of phase diffs, starting at strtpair
for (i,pair) in enumerate(pairs[strtpair:]): 
    ax1.plot(freq,coh[pair],label='coh'+str(pair),color=colorset[i],linewidth=lw)
    ax2.plot(freq,cp[pair],color=colorset[i],label='dph'+str(pair), linestyle=['--','-'][i==0 and strtpair==0],
            linewidth=lw)
    

if ax1==ax2:
    plt.setp(ax1, ylim=(-4,4), ylabel = 'phase difference (rad) || coherence')
    ax1.legend(prop={'size':'x-small'},loc='lower right',ncol=2)
else:
    plt.setp(ax1, ylim=(-.5,1.1), ylabel = 'coherence')
    plt.setp(ax2, ylim=(-4,4), ylabel = 'phase difference (rad)')
    [ax.legend(prop={'size':'x-small'},loc='lower right',ncol=2) for ax in [ax1,ax2]]
    
totph=np.sum([cp[pair] for pair in pairs[1:]],0)
hist = data.history
sz = 'small' if len(hist.split('\n'))>3 else 'medium'
ax1.set_title(hist + str(': NFFT={NFFT}, noverlap={noverlap}'
                        .format(NFFT=NFFT, noverlap=noverlap)),size=sz)
ax3.plot (freq,totph,'--',label='total phase',color='b',linewidth=lwtot)
# plot a few to make sure at least one is in range
for offs in [0, 2]:
    ax3.plot(freq, mod(totph,2*pi)-offs*pi,':',color='b',linewidth=lw)
endendpair = pairs[0]
for offs in [0,1,2]:
    ax3.plot(freq,cp[endendpair]-offs*2*pi,color='r',
             linewidth=lwtot,label=[endendpair,'',''][offs])
plt.setp(ax3, xlabel=str('frequency ({xlab})'.format(xlab=xlab)), xlim = [1,100])   
ax3.legend(prop={'size':'small'},loc='lower right')

plt.show(block=0)

