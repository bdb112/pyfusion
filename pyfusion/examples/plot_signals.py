""" see plot_signal_trivial for bare bones 
plots a single or multi-channel signal
This example shows bad motorboating
run pyfusion/examples/plot_signals dev_name='H1Local' diag_name='H1Poloidal1' shot_number=76887
# multichannel example
run pyfusion/examples/plot_signals dev_name='H1Local' diag_name='ElectronDensity' shot_number=76887
run pyfusion/examples/plot_signals diag_name='LHD_n_e_array' shot_number=42137 sharey=1
# npz local files
run pyfusion/examples/plot_signals.py dev_name='HeliotronJ' shot_number=50136 diag_name=HeliotronJ_MP_array
# Okada style files
run pyfusion/examples/plot_signals.py dev_name='HeliotronJ' shot_number=58000 diag_name=HeliotronJ_MP_array
"""
import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt

_var_defaults = """
dev_name = "LHD"
diag_name = "MP1"
shot_number = 27233
sharey=False
fun=myiden
fun2=myiden2
plotkws={}
hold=0
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)

if hold==0: plt.figure()

# maybe should be in data/plots.py, but config_name not fully implemented
data.plot_signals(suptitle='shot {shot}: '+diag_name, sharey=sharey,
                  fun=fun, fun2=fun2, **plotkws)

# for shot in range(76620,76870,10): dev.acq.getdata(shot,diag_name).plot_signals()
