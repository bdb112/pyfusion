import pylab as pl
import numpy as np
from pyfusion.visual import sp, vis    
from pyfusion.acquisition.LHD.read_igetfile import igetfile

pl.rcParams['legend.fontsize']='small'


_var_default = """
egdiags = 'wp,ip'
shot=105396
diag_names='VSL0011'
diag_names='VSL_SMALL'
dev_name='LHD'
offset = 0    # -3  # for VSL signals before implementing pretrig
hold = 0  # -1 for figure()
"""
import pyfusion.utils
exec(_var_default)
exec(pyfusion.utils.process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
if hold == -1: pl.figure()
elif hold == 0: pl.clf()

for diag in diag_names.split(','):
    data = dev.acq.getdata(shot,diag)
    if offset is not None:
        data.timebase += offset
    data.plot_signals(labeleg='True')

for eg in egdiags.split(','):
    egdat = igetfile('{d}@{s}.dat'.format(s=shot, d=eg))
    egdat.plot(1)

