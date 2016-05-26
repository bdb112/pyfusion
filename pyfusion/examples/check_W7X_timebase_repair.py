""" from plot_signal.py
This runs some checks on W7X signals with corrupt timebases
The first (so far only) method is to check the voltage data for time slips by using analytic_phase
"""
import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
from pyfusion.data.signal_processing import analytic_phase
from numpy import pi
from pyfusion.utils import fix2pi_skips, modtwopi
# fix2pi_skips  seems to slow things down.

_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_L57_LP1_8"
shot_number = [20160224,3] 
sharey=True
plotkws={}
fsamp=498.94
def aphase(t,y): return (t,analytic_phase(y)-fsamp*2*pi*t)  # no fooling with time
# replace time vector with a faked time
def aphase_fix_time(t,y): tt=2e-6*arange(len(t)); return(tt,analytic_phase(y)-498.94*2*pi*tt)
fun=myiden
fun2=aphase
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)
print('data length is ',len(data.timebase))
fd = data
#fd = data.filter_fourier_bandpass(passband=[490,510], stopband=[450,550])
print('filtered data length is ',len(fd.timebase))

# maybe should be in data/plots.py, but config_name not fully implemented
fd.plot_signals(suptitle='shot {shot}: '+diag_name + ' fsamp=' + str(round(fsamp,2)), sharey=sharey,
                  fun=fun, fun2=fun2, **plotkws)

