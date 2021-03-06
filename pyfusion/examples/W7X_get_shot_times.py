""" start to compare times between the progiD, Scenario DB and diagnostics
So far this works on cached files, but when it is developed it will need to
access the archiveDB

_PYFUSION_TEST_@@Skip@@  only works on W7X net
"""


import pyfusion
from pyfusion.acquisition.W7X.get_shot_info  import get_shot_utc
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt
import numpy as np

_var_defaults = """
dev_name = "W7X"
ECH = "W7X_TotECH"
shot = [20160310,9]
plotkws={}
hold=0
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
ECHdata = dev.acq.getdata(shot, ECH)
t0,t_end = get_shot_utc(shot)
toffs_ECH = 60-(ECHdata.utc[0]-t0)/1e9

wech = np.where(ECHdata.signal > 100)[0]
tech = ECHdata.timebase[wech[0]]
utc0 = int(tech * 1e9) + ECHdata.utc[0]

first_ax = None
plt.plot(ECHdata.timebase - toffs_ECH, ECHdata.signal)
