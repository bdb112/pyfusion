import pyfusion
import matplotlib.pyplot as plt
import numpy as np

from pyfusion.utils import process_cmd_line_args
from pyfusion.data.shot_range import shot_range
#_PYFUSION_TEST_@@"shot_list=shot_range([20160310,11],[20160310,15])"
_var_defaults = """
shot_list = [[20160310,i] for i in range(7,12)]  # can be a list of lists e.g. grouped shots
diag = 'W7X_TotECH'
dev_name = 'W7X'
plot_sig_kws = dict()
nrows=None  #  3
ncols=None  #  4
t_range = []
sharex='all'
sharey='all'
plot_fun='plot_signals'  # or 'plot_specgram'
# To control the number ofx ticks 
#   gca().locator_params(nbins=4, axis='x')
"""
exec(_var_defaults)
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)

i = 0
axs = []
row_widths = [len(r) for r in shot_list if np.shape(r) is not ()]
if len(row_widths) > 0 and shot_list[0][0] < 20160000:
    print('Detected a grouped shot list - matching ncols and nrows to it.')
    ncols = np.max(row_widths)
    nrows = len(shot_list)
    padded_list = []
    for r in range(nrows):
        padded_row = ncols*[None]
        for c, sh in enumerate(shot_list[r]):
            padded_row[c] = sh
        padded_list.extend(padded_row)

    shot_list = padded_list
    
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
    if shot is None: continue
    if i >= len(axs):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=[12, 8], sharex=sharex, sharey=sharey)
        axs = axs.flatten()
        fig.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.95, wspace=.1+0.2/ncols, hspace=0.15+0.2/nrows)

        i = 0

    plt.sca(axs[i])
    try:
        data = dev.acq.getdata(shot, diag)
        if data is None:
            if None in shot_list: # stick with the grid if we have grouped shots
                i += 1 
            continue
        if len(t_range)>0:
            data=data.reduce_time(t_range)
        
        if plot_fun == 'plot_signals':
            data.plot_signals(suptitle='', **plot_sig_kws)
        else:
            data.plot_spectrogram(suptitle='', title=' ', **plot_sig_kws)
        axs[i].set_title(str(shot),loc='left', fontsize='large')
        maxwid = 10 if ncols>4 else 16
        axs[i].set_title(diag[-maxwid:],loc='right', fontsize='small')
        i += 1
    except LookupError:
        pass

plt.show(block=0)
