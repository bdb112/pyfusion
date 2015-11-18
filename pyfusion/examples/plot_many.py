""" plot a single diagnostic for a range of shots.  Could be made a subset of plot_multi?
"""
import pyfusion
import pylab as pl
import numpy as np
from pyfusion import H1_scan_list

all_shots = H1_scan_list.get_all_shots()


_var_defaults = """
dev_name = "H1Local"
diag_name = "H1PoloidalMirnov1_6"
shot_list = all_shots[1:1000:10]
rows=3
cols=4
exception=Exception
ylims=None
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

bad = []

sharey=None
if ylims is not None:
    sharey='all'

dev = pyfusion.getDevice(dev_name)
fig, axs = pl.subplots(rows, cols, sharex='all',sharey=sharey)
print len(zip(axs.flatten(),shot_list))
for ax,shot in zip(axs.flatten(),shot_list):
    print shot,
    try:
        data = dev.acq.getdata(shot,diag_name)
        pl.sca(ax)
        #data.plot_signals()
        y = data.signal
        t = data.timebase
        
        ax.plot(t[0:len(y)],y)
        ax.set_title(shot)
    except exception:
        bad.append(shot)
if ylims is not None:
    ax.set_ylim(ylims)

#fig.tight_layout(pad=.2) # seems to need a 'show' after resizing window
fig.suptitle(diag_name)
fig.subplots_adjust(left=.02, right=0.98, bottom=0.02, top=0.95, wspace=.2, hspace=0.2)
pl.show()
        
print len(bad),' bad shots (in bad)'
