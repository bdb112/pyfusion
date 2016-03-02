""" from plot_signal.py

"""
import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
from pyfusion.data.signal_processing import analytic_phase
from numpy import pi

_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_L57_LP1_8"
shot_number = [20160224,3] 
sharey=True
plotkws={}
fsamp=498.94
def aphase(t,y): return(t,analytic_phase(y)-498.94*2*pi*t)  # no fooling with time
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
fd = data.filter_fourier_bandpass(passband=[490,510], stopband=[450,550])
print('filtered data length is ',len(fd.timebase))

# maybe should be in data/plots.py, but config_name not fully implemented
fd.plot_signals(suptitle='shot {shot}: '+diag_name, sharey=sharey,
                  fun=fun, fun2=fun2, **plotkws)

