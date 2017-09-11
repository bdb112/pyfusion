import pyfusion
import matplotlib.pyplot as plt
import numpy as np

from pyfusion.utils import process_cmd_line_args
from pyfusion.data.shot_range import shot_range
#_PYFUSION_TEST_@@"shot_list=shot_range([20160310,11],[20160310,15])"
_var_defaults = """
shot_list = [[20160310,i] for i in range(7,12)]
diag = 'W7X_TotECH'
dev_name = 'W7X'
plot_sig_kws = dict()
nrows=None  #  3
ncols=None  #  4
t_range = []
sharex='all'
"""
exec(_var_defaults)
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)

i = 0
axs = []
if nrows is None and ncols is None:
    if len(shot_list) <= 4:
        nrows = len(shot_list)
        ncols = 1
    else:
        nrows = int(np.sqrt(len(shot_list)) * 1.2)
        ncols = int(np.sqrt(len(shot_list)) * 1.4)
elif nrows is None:
    nrows = 1+ncols
    
for shot in shot_list:
    if i >= len(axs):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=[12, 8], sharex=sharex)
        axs = axs.flatten()
        fig.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.90, wspace=.35, hspace=0.2)
        i = 0

    plt.sca(axs[i])
    try:
        data = dev.acq.getdata(shot, diag)
        if len(t_range)>0:
            data=data.reduce_time(t_range)
        data.plot_signals(suptitle='', **plot_sig_kws)
        axs[i].set_title(str(shot))
        i += 1
    except LookupError:
        pass

plt.show(block=0)
