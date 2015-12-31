"""
Plot coherence and cross phase for a multi signal diagnostic
Can pass a pyfusion data item or get data from a source
H-1 88673 is a nice example

mlab.cohere_pairs is > 10x faster on 15 signals than separate cohere() calls
See also csd in  /usr/lib/pymodules/python2.7/matplotlib/pyplot.py
The fully annotated version of this is in examples/JSPF_tutorial
"""

import pyfusion
import pylab as pl
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
strtpair=0
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

xlab = {1:'Hz', 1000:'kHz', 1000000:'MHz'}[unit_freq] 

if type(noverlap) != int:
    noverlap = int(noverlap*NFFT)

if dat is None:
    dev = pyfusion.getDevice(dev_name)
    orig_data = dev.acq.getdata(shot_number,diag_name)
    data = orig_data
else:
    data = dat

if time_range[1]!=time_range[0]:
    data = data.reduce_time(time_range)

dt = np.average(np.diff(data.timebase))

X=np.array([s for s in data.signal]).T

if chans is None: 
    chans = [i for i in range(np.shape(X)[1]-1)]
for o in omit:
    chans.remove(o)

pairs = [(chans[i],chans[i+1]) for i in range(len(chans)-1)]

pairs.insert(0, (pairs[0][0],pairs[-1][1]))  #  the first elt is from end to end
coh,cp,freq = pl.mlab.cohere_pairs(X,pairs,NFFT=NFFT,Fs=1/dt/float(unit_freq))

colorset=('b,g,r,c,m,y,k,orange,purple,lightgreen,gray,yellow,brown,teal,tan'.split(',')) # to be rotated

ax = pl.gca()
for (i,pair) in enumerate(pairs[strtpair:]): 
    ax.plot(freq,coh[pair],label='dph'+str(pair),color=colorset[i],linewidth=lw)
    ax.plot(freq,cp[pair],color=colorset[i],linestyle=['--','-'][i==0 and strtpair==0],
            linewidth=lw)
    
pl.setp(ax, ylim=(-4,4), ylabel = 'phase difference (rad) || coherence')
totph=np.sum([cp[pair] for pair in pairs[0:-1]],0)
hist = data.history
sz = 'small' if len(hist.split('\n'))>3 else 'medium'
ax.set_title(hist + str(': NFFT={NFFT}, noverlap={noverlap}'
                        .format(NFFT=NFFT, noverlap=noverlap)),size=sz)
ax.plot (freq,totph,':',label='total phase',color='b',linewidth=lwtot)
# plot a few to make sure at least one is in range
for offs in [0, 2]:
    ax.plot(freq, mod(totph,2*pi)-offs*pi,':',color='b',linewidth=lwtot)
endendpair = pairs[0]
for offs in [0,1,2]:
    ax.plot(freq,cp[endendpair]-offs*2*pi,color='r',
             linewidth=lwtot,label=[endendpair,'',''][offs])
pl.setp(ax, xlabel=str('frequency ({xlab})'.format(xlab=xlab)), xlim = [1,100])   
ax.legend(prop={'size':'small'},loc='lower right')

pl.show(block=0)

