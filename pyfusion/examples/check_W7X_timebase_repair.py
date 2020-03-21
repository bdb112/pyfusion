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
sharex=True
plotkws={}
fsamp=498.94
iterate=1
def aphase(t,y): return (t,analytic_phase(y)-fsamp*2*pi*t)  # no fooling with time - but the fsamp is 'hardwired' - see aphase_dyn below
# replace time vector with a faked time
def aphase_fix_time(t,y): tt=2e-6*arange(len(t)); return(tt,analytic_phase(y)-498.94*2*pi*tt)
fun=myiden
fun2=aphase
add_corruption=0  # add three dislocations a long, short (~3ms) and vshort 0.5ms
fft=1
block=0
time_range=None
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
if shot_number[0] == 20160310:
    pyfusion.utils.warn(' need to restore sin ')
data = dev.acq.getdata(shot_number,diag_name, time_range=time_range)

print('data length is ',len(data.timebase))
if add_corruption:
    for sig in data.signal:  # small dislocation for 0309_52
        sig[100000:140000] = sig[140001:180001]  # small disloc (depends on fsw)
        sig[200000:201450] = sig[201450:202900]  # big dislocation 1.5 cycles
        sig[300000:300450] = sig[300450:300900]  # small 0.5 cycles
fd = data
# this is a workaround for clipped sweep signals, but it slows time response
if fft:
    FFT_size = nice_FFT_size(len(data.timebase)-1, -1) # 1 less to allow rounding err in reduce time
    data = data.reduce_time([data.timebase[0], data.timebase[FFT_size]])
 
    fd = data.filter_fourier_bandpass(passband=[100,900], stopband=[50,950])
    print('filtered data length is ',len(fd.timebase))

phc = analytic_phase(data[data.keys()[0]]) - fsamp*2*pi*data.timebase
ends = len(phc)/10
#fsamp += np.diff(phc)[ends:-ends].mean()/np.diff(data.timebase).mean()/(2*pi)
if iterate:  # this correction looks just at data between 30% and 70% to avoid hiccups from shot with no plasma
    dfsamp = (phc[-3*ends:-2*ends].mean() - phc[2*ends:3*ends].mean())/(
        (data.timebase[-3*ends:-2*ends].mean() - data.timebase[2*ends:3*ends].mean()))/(2*np.pi)
    if np.isnan(dfsamp):
        print('*** Unable to correct fsamp due to Nans - find the best fsamp on a shorter segment')
    else:
        fsamp += dfsamp
        
        # this line refines the aphase fun used in plot_signal call
exec("def aphase_dyn(t,y): print('recompiled', fsamp); return (t,analytic_phase(y)-fsamp*2*pi*t)")

# maybe should be in data/plots.py, but config_name not fully implemented
# note that this layout is overridden by a multi diag
fig, [axph, axdt] = plt.subplots(2, 1)
plt.sca(axph)
fd.plot_signals(suptitle='shot {shot}: '+diag_name + ' fsamp=' + str(round(fsamp,2)),
                sharey=sharey, sharex=sharex, hold=1, fun=fun, fun2=aphase_dyn, **plotkws)

# compare timebase from signal with one reconstructed from the rawdim, if present
# due to bugs, it only works with single diagnostics
if 'W7X' in data.keys()[0]:
    if len(np.shape(data.signal))>1:
        print('raw timebase check only works with single channel diags at present')
    else:
        dt = data.params.get('diff_dimraw', None)
        if dt is None:
            raise Exception(' no diff_dimraw found in params')
        biggest_float = np.float128 if hasattr(np, 'float128') else np.float64
        t_zero = (data.params['req_f_u']-data.params['utc_0'])/1e9
        offs = dt[0]
        dt[0] = 0
        tbcheck = np.clip(np.cumsum(dt.astype(biggest_float)), 0, None)/1e9
        axdt.set_title('diff of dimraw in secs')
        axdt.plot(data.timebase, tbcheck[0:len(data.timebase)] + t_zero - data.timebase)
        axdt.set_yscale('symlog', linthreshy=1e-6)
plt.show(block)
if abs(np.diff(axph.get_ylim()))>10:
    print('***\n***Failure to reduce slope may be due to shift in input signal level > some amplitude (such as before plasma).  Signal is sampled between 30% and 70% time')
