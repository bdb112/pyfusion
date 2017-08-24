""" plot correlation of two diagnostics over a time range
"""
import pyfusion
import matplotlib.pyplot as plt
from pyfusion.data.pyfusion_corrinterp import correlation

_var_defaults = """
dev_name = "H1Local"
diag1 = 'H1ToroidalAxial'
diag2 = 'H1ToroidalMirnov_15y'
shot_number = 100617
# shot_number = [20160308,39]
# dev_name = "LHD"; diag_name = "MP1" ; shot_number = 27233
t_range = []
hold = 0
coefft = 1
bp = [] # tuple [freq (Hz), bw(rel to freq), order (not implemented)]  - if longer, use as a digital filter
"""
exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data1 = dev.acq.getdata(shot_number, diag1)
data2 = dev.acq.getdata(shot_number, diag2)

if len(t_range) > 0:
    data1 = data1.reduce_time(t_range)
    data2 = data2.reduce_time(t_range)

if hold==0: plt.figure()

plt.plot(correlation(data1, data2, coefft=coefft))

plt.show()
