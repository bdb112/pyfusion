""" This was practice for a subroutine to restore the clipped part of the
sinusoid, now in data/restore_sin.py (for a single channel)
""" 

import numpy as np
from scipy.fftpack import hilbert
import matplotlib.pyplot as plt
import pyfusion
from pyfusion.debug_ import debug_

def analytic_signal(x):
    """ A short-cut assuming that the incoming signal is reasonable
    e.g. fairly pure sinusoid.
    """ 
    x = x - np.mean(x)
    return(x+1j*hilbert(x))

_var_defaults = """
verbose=1
dev_name = "W7X"
diag_name = "W7X_L57_LP01"
shot_number = [20160310,9] 
sharey=True
t_range=[1.2,1.4]
method = 2
sweep_freq =  500  # sinusoidal sweep freq in Hz
Vpp = 90*2  # pp value of signal before it was clipped 
clip_level_minus = -88  # value to ensure even soft clipping is excluded
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)
rd = data.reduce_time(t_range)

if verbose>0:
    rd.plot_signals()
# fourier filter to retain the fundamental - the amplitude with be reduced by the clipping
stopband = sweep_freq * np.array([0.8,1.2])
passband = sweep_freq * np.array([0.9,1.1])
fd=rd.filter_fourier_bandpass(stopband=stopband,passband=passband)

# calculate the time-varying amplitude of the filtered sinusoid
amp = np.abs(analytic_signal(fd.signal[0]))/np.sqrt(2)
# normalise to one
amp = amp/np.average(amp)
fig, ax1 = plt.subplots(1,1)
if verbose>0: 
    ax1.plot(rd.timebase, rd.signal[0],'b', label = 'orig',linewidth=.3)
if verbose>1:
    ax1.plot(rd.timebase, fd.signal[0],'g',label = 'filtered',linewidth=.3)
if method==1:
    ax1.plot(rd.timebase, fd.signal[0]/amp,'m', label='corrected',linewidth=.3)
    ax1.plot(rd.timebase, 50*(1.3*amp-2.1) + 1.2*fd.signal[0]/amp,'r', label='corrected')

# not bad, but try making the amplitude constant first, then take the
# difference , excluding the clipped part, boxcar averaged over a
# small, whole number of periods

for i in range(2):  # iterate to restore amplitude to a constant
# first the reconstructed amplitude
    reconst = 1.0 * fd.signal[0] # make a copy of the signal
    amprec = np.abs(analytic_signal(reconst))
    reconst = Vpp/2.0 * reconst/amprec
if method==2:
    plt.plot(rd.timebase, reconst,'m', label='reconst before DC adjust')

# should have a very nice constant ampl. sinusoid
# now blank out the clipped, use given value because amplifier clipping 
# is 'soft', so automatic detection of clipping is not simple.
wc = np.where(rd.signal[0] < clip_level_minus)[0]
weight = 1 + 0*reconst
weight[wc] = 0
period = 1000
from pyfusion.data.signal_processing import smooth
# iterate to make waves match where there is no clipping
for i in range(6):
    err = rd.signal[0] - reconst
    err[wc] = 0
    print(sum(err)/sum(weight))
    corrn = np.cumsum(err[0:-period]) - np.cumsum(err[period:])
    divisor = np.cumsum(weight[0:-period]) - np.cumsum(weight[period:]) 
    wnef = np.where(divisor <= 100)[0]
    divisor[wnef] = 100
    if verbose>1:
        ax1.plot(rd.timebase, reconst, '--', 
                 label='reconst, offset {i}'.format(i=i))
    # reconst[period//2:-period//2] = reconst[period//2:-period//2] - corrn/divisor
    reconst[period//2:1-period//2] = reconst[period//2:1-period//2] + smooth(err,period)/smooth(weight,period)
    # plot(smooth(err,period)/smooth(weight,period))

    debug_(pyfusion.DEBUG, 1, key='restore_sin')
if method==2:
    if verbose>0:
        ax1.plot(rd.timebase, reconst,'r', label='reconst, final offset')
    
ax1.legend()
fig.show()


