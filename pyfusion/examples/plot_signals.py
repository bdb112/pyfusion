""" see plot_signal_trivial for bare bones """
import pyfusion
_var_defaults = """
dev_name = "LHD"
chan_name = "MP1"
shot_number = 27233
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

h1=pyfusion.getDevice(dev_name)
data=h1.acq.getdata(shot_number,chan_name)
data.plot_signals()
