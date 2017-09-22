import pyfusion
import matplotlib.pyplot as plt
import numpy as np
from pyfusion.data.shot_range import shot_range
from pyfusion.utils import process_cmd_line_args

_var_defaults = """
shot_list = [[20160310,i] for i in range(7,12)]  # can be a list of lists e.g. grouped shots
shot_list = shot_range(97419, 97427)
diag = 'H1ToroidalAxial'
dev_name = 'H1Local' 
plot_sig_kws = dict()
xlim=[0.01,0.083]
ylim=[0,100]
clim=[-90,0]
dir = '/tmp/movie'
figsize = [12, 12]
"""

exec(_var_defaults)
exec(process_cmd_line_args())

plt.rc('font', size=8)
plt.figure(figsize=figsize)

for shot in shot_list:
    dev = pyfusion.getDevice(dev_name)
    data = dev.acq.getdata(shot, diag)
    if data is None:
        continue
    data.plot_spectrogram(xlim=xlim, ylim=ylim, clim=clim, **plot_sig_kws)
    plt.gca().locator_params(nbins=5, axis='x')
    plt.subplots_adjust(left=0.07, bottom=0.06, right=0.96, top=0.96)
    plt.savefig(dir + '/{diag}_{shot}'.format(diag=diag, shot=shot))
