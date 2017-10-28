""" from plot_signal.py
This runs some checks on W7X signals with corrupt timebases
The first (so far only) method is to check the voltage data for time slips by using analytic_phase

_PYFUSION_TEST_@@diag_name=W7X_L57_LP01_U shot_number=[20160309,51] fsamp=500.41 fft=0 block=0
"""
import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
from pyfusion.data.signal_processing import analytic_phase
from numpy import pi
import numpy as np
from matplotlib import pyplot as plt
from pyfusion.utils import fix2pi_skips, modtwopi
from pyfusion.data.restore_sin import restore_sin
from pyfusion.utils.primefactors import nice_FFT_size

# fix2pi_skips  seems to slow things down.

_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_L57_LP01_08"
shot_number = [20160224,3] 
sharey=True
plotkws={}
fsamp=498.94
def aphase(t,y): return (t,analytic_phase(y)-fsamp*2*pi*t)  # no fooling with time - but the fsamp is 'hardwired'
# replace time vector with a faked time
def aphase_fix_time(t,y): tt=2e-6*arange(len(t)); return(tt,analytic_phase(y)-498.94*2*pi*tt)
fun=myiden
fun2=aphase
add_corruption=0  # add three dislocations a long, short (~3ms) and vshort 0.5ms
fft=1
block=0
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
if shot_number[0] == 20160310:
    pyfusion.utils.warn(' need to restore sin ')
data = dev.acq.getdata(shot_number,diag_name)
print('data length is ',len(data.timebase))
if add_corruption:
    for sig in data.signal:  # small dislocation for 0309_52
        sig[100000:140000] = sig[140001:180001]  # small disloc (depends on fsw)
        sig[200000:201450] = sig[201450:202900]  # big dislocation 1.5 cycles
        sig[300000:300450] = sig[300450:300900]  # small 0.5 cycles
fd = data
# this is a workaround for clipped sweep signals, but it slows time response
if fft:
    FFT_size = nice_FFT_size(len(data.timebase), -1)
    data = data.reduce_time([data.timebase[0], data.timebase[FFT_size]])
 
    fd = data.filter_fourier_bandpass(passband=[100,900], stopband=[50,950])
    print('filtered data length is ',len(fd.timebase))

phc = analytic_phase(data.signal[0]) - fsamp*2*pi*data.timebase
fsamp += np.diff(phc).mean()/np.diff(data.timebase).mean()/(2*pi)


# maybe should be in data/plots.py, but config_name not fully implemented
fd.plot_signals(suptitle='shot {shot}: '+diag_name + ' fsamp=' + str(round(fsamp,2)), sharey=sharey,
                  fun=fun, fun2=fun2, **plotkws)

# compare timebase from signal with one reconstructed from the rawdim, if present
# due to bugs, it only works with single diagnostics
if 'W7X' in data.keys()[0]:
    if len(np.shape(data.signal))>1:
        print('raw timebase check only works with single channel diags at present')
    else:
        dt = data.params.get('diff_dimraw', None)
        if dt is None:
            raise Exception(' no diff_dumraw found in params')
        tbcheck = np.clip(np.cumsum(dt.astype(np.float128))-dt[0], 0, None)/1e9
        plt.figure()
        ax = plt.gca()
        ax.set_title('diff of dimraw')
        ax.plot(tbcheck - data.timebase)
        ax.set_yscale('symlog', linthreshy=1e-6)
plt.show(block)










