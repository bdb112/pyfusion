import pyfusion
import numpy as np
from matplotlib import pyplot as plt
"""
should separate out into something like prl_python_h1.h1 import write_to_summary
_PYFUSION_TEST_@@db_url="None"
"""

_var_defaults = """
dev_name = 'H1Local'
diag_name = 'H1_Diamag'
shots = [102266]
bl = [-0.05, 0, 0.04, 0.09]
dbg = 3  # if > 0, the number of shots to plot
db_url='sqlite:////data/summary.db' # or 
#db_url='sqlite:////rmt/h1svr/home/datasys/virtualenvs/h1ds_production/h1ds/h1ds/db/summary.db' 
# use "None" (quoted string) to stop db access
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if db_url !='None':
    from sqlsoup import SQLSoup

    def write_to_summary(shot, valdic):
        if not isinstance(shot, (tuple, list, np.ndarray)):
            shot = [shot]
        for s in shot:
            SUMM.filter(h1db.summary_h1.shot == s).update(valdic)


    h1db = SQLSoup(db_url)
    cols = h1db.summary_h1.c.keys()
    SUMM = h1db.summary_h1
else:
    h1db = None
    
bads, dias, IPs = [], [], []


for i, shot in enumerate(shots):
    dev = pyfusion.getDevice(dev_name)
    try:
        dat = dev.acq.getdata(shot, diag_name)
    except:
        bads.append(shot)
        continue

    if dat.signal[0] > -10 or dat.signal[-1] > -30:
        IPs.append(shot)
        continue
    dias.append(shot)
    datr = dat.reduce_time([bl[0], bl[3]], copy=True)
    datflat = datr.remove_baseline(baseline=bl)
    datint = datflat.integrate(baseline=None)
    dI = datint.reduce_time(bl[2:]).signal.mean() - datint.reduce_time(bl[:2]).signal.mean()
    dia = float(dI)/(bl[2]-bl[1])
    if (dbg > 0) and (len(dias) < dbg):
        if len(dias) == 1:
            ax1 = plt.gca()
            ax2 = ax1.twinx()
        plt.sca(ax1)
        datflat.plot_signals(color=None, labeleg='True', labelfmt='')
        plt.sca(ax2)
        datint.plot_signals(color='r')
    print('{s} {d:.3f}'.format(s=shot, d=dia))
    if h1db is not None:
        write_to_summary(shot, dict(test=dia))

ax1.legend()
if h1db is not None:
    h1db.commit()
