"""
Plot coherence and cross phase for a multi signal diagnostic
Can pass a pyfusion data item or get data from a source

mlab.cohere_pairs is > 10x faster on 15 signals than separate coheres 
See also csd in  /usr/lib/pymodules/python2.7/matplotlib/pyplot.py

fdata=data.filter_fourier_bandpass([20e3,36e3],[24e3,32e3])
f0,f1=fdata.signal[[3,4]];plot(smooth_n(f0*f1,n)/sqrt(smooth_n(f0**2,n)*smooth_n(f1**2,n)));plot(f0,'b');plot(f1,'g');
"""
""" see plot_signal_trivial for bare bones 
plots a single or multi-channel signal"""
import pyfusion
import pylab as pl
import numpy as np
from numpy import mod, pi

_var_defaults = """
dev_name = "H1Local"           # "LHD"
diag_name = "H1ToroidalAxial"  # "MP1"
shot_number = 88675            # 27233
lwtot=1.5
NFFT=256
noverlap=1.0
unit_freq = 1000
dat=None
time_range=[0,0]
lw=1
chans=None
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

#%run pyfusion/examples/plot_signals shot_number=88672 diag_name='H1ToroidalAxial' dev_name='H1Local'
pl.figure()
X=np.array([s for s in data.signal]).T

if chans is None: 
    chans = [i for i in range(np.shape(X)[1]-1)]
pairs = [(chans[i],chans[i+1]) for i in range(len(chans)-1)]

pairs.insert(0, (pairs[0][0],pairs[-1][1]))
coh,cp,freq = pl.mlab.cohere_pairs(X,pairs,NFFT=NFFT,Fs=1/dt/float(unit_freq))

colorset=('b,g,r,c,m,y,k,orange,purple,lightgreen,gray,yellow,brown,teal,tan'.split(',')) # to be rotated

for (i,pair) in enumerate(pairs): 
    pl.plot(freq,coh[pair],label=pair,color=colorset[i],linewidth=lw)
    pl.plot(freq,cp[pair],color=colorset[i],linestyle=['--','-'][i==0],linewidth=lw)
    
pl.ylim(-4,4)
totph=np.sum([cp[pair] for pair in pairs[0:-1]],0)
pl.plot (freq,totph,label='total phase',color='b',linewidth=lwtot)
pl.ylim(-12,4)
pl.plot(freq, mod(totph,2*pi),color='b',linewidth=lwtot)
pl.plot(freq, mod(totph,2*pi)-2*pi,color='b',linewidth=lwtot)
pl.legend(prop={'size':'small'},loc='lower right')
hist = data.history
sz = 'small' if len(hist.split('\n'))>3 else 'medium'
pl.title(hist + str(': NFFT={NFFT}, noverlap={noverlap}'
                            .format(NFFT=NFFT, noverlap=noverlap)),size=sz)
pl.xlabel(xlab)
pl.show()



