import pyfusion as pf
import pylab as pl
import numpy as np

# now that get_basic is renamed and in LHD:
from pyfusion.acquisition.LHD.get_basic_diagnostics import get_basic_diagnostics as get_basic_params

_var_defaults="""
dev_name = "LHD"
shot=38075
diags = ['<n_e19>', 'w_p', 'b_0']
times = np.linspace(0,4,1000)
delay=None
hold=1
scale=1
debug=0
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

device = pf.getDevice(dev_name)

#data=dev.acq.getdata(shot,diag_name)
#data.plot_signals()

bp = get_basic_params(diags,shot=shot,times=times,delay=delay,debug=debug)
junk = bp.pop('check_tm')
junk = bp.pop('check_shot')


for (i,k) in enumerate(bp.keys()):
    y = bp[k]
    label=k
    if label == '<n_e19>':
        label = '<n_e17>'
        y=100*y
    elif label == 'b_0':
        label = '100x'+label
        y=100*y

    if np.average(y)<0:
        label = '-'+label
        y = -y

    if scale != 1:
        label = '{0:.1g}*'.format(scale)+label

    if (hold == 0) and (i == 0) : 
        pl.plot(hold=0)

    pl.plot(times, scale*y, label = label)
    pl.legend()
    pl.title("{s}: {b_0:.3g}T".format(s=shot, b_0=bp['b_0'][0]))
pl.show(block=0)
 
