import pyfusion
from pyfusion.acquisition.W7X.get_shot_utc  import get_shot_utc
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt

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
t0,t_end = get_shot_utc(*shot)
 
toff_ECH = 60-(ECHdata.utc[0]-t0)/1e9


plt.plot(ECHdata.timebase - toff_ECH, ECHdata.signal)
