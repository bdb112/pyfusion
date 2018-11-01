"""
create a simple table in simple array
_PYFUSION_TEST_@@shot_range='get_shot_range([20180911,24], [20180911,26])'

"""

from __future__ import print_function
import pyfusion
import numpy as np       
import matplotlib.pyplot as plt
from pyfusion.data.shot_range import shot_range as get_shot_range
import sys

sys.path.append('/home/bdb112/python')
from pyfusion.utils.process_cmd_line_args import process_cmd_line_args


_var_defaults = """
dev_name = 'W7M'
stop = 0
diag_name = 'W7M_BRIDGE_VDIFF'
shot_range = get_shot_range([20180911,24], [20180912,99])
time_range = [.9, 1]
"""

def __help__():  # must be before exec() line
    print(__doc__)
    print('local help routine for sqlplot!')

exec(_var_defaults)
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)

table = []
for shot in shot_range:
    dat = dev.acq.getdata(shot, diag_name, contin=not stop)
    if dat is None:
        continue

    if time_range is not []:
        dat.reduce_time(time_range, copy=False)

    indmaxall = np.argmax(dat.signal)
    ind95 = np.argsort(dat.signal)[int(len(dat.signal) * 0.98)]
    max95 = dat.signal[ind95]
    maxall = dat.signal[indmaxall]
    if 'VDIFF' in diag_name:
        max95, maxall = [20 * x  for x in [max95, maxall]]
    info = (shot, dat.timebase[ind95], max95, dat.timebase[indmaxall], maxall)
    table.append(info)
    print(info)

for info in table:
    print('{shot:15s}: {rest}'.format(shot=str(info[0]), rest = '   '.join(['{0:8.4f}'.format(x) for x in info[1:]])))
