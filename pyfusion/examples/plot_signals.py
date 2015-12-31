""" see plot_signal_trivial for bare bones 
plots a single or multi-channel signal
This example shows bad motorboating
run pyfusion/examples/plot_signals dev_name='H1Local' diag_name='H1Poloidal1' shot_number=76887
# multichannel example
run pyfusion/examples/plot_signals dev_name='H1Local' diag_name='ElectronDensity' shot_number=76887
run pyfusion/examples/plot_signals diag_name='LHD_n_e_array' shot_number=42137 sharey=1
run pyfusion/examples/plot_signals.py dev_name='HeliotronJ' shot_number=50136 diag_name=HeliotronJ_MP_array
"""
import pyfusion

_var_defaults = """
dev_name = "LHD"
diag_name = "MP1"
shot_number = 27233
sharey=False
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)

# maybe should be in data/plots.py, but config_name not fully implemented
data.plot_signals(suptitle='shot {shot}: '+diag_name, sharey=sharey)

# for shot in range(76620,76870,10): dev.acq.getdata(shot,diag_name).plot_signals()
