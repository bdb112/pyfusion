"""
Extract the diamagnetic signal using careful basleine removal, and put into db

should separate out into something like prl_python_h1.h1 import write_to_summary
_PYFUSION_TEST_@@db_url="None"
"""

import pyfusion
import numpy as np
from matplotlib import pyplot as plt
from pyfusion.data.signal_processing import smooth

_var_defaults = """
dev_name = 'H1Local'
diag_name = 'H1_Diamag'
shots = [102266]
bl=[-0.05,0,0.1,0.19]  # for finding the rise and falls
dt=0.05  # for final baseline (corrected baseline, and then for integral)
tfilt = 0.002 # time constant for filtering before looking for rise and fall
dbg = 3  # if > 0, the number of shots to plot
db_url='sqlite:////data/summary.db' # or 
#db_url='sqlite:////rmt/h1svr/home/datasys/virtualenvs/h1ds_production/h1ds/h1ds/db/summary.db' 
# use "None" (quoted string) to stop db access
debug=1
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
    
bads, simple, best, IPs = [], [], [], []


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

    datr = dat.reduce_time([bl[0], bl[3]], copy=True)
    # first baseline removal to get rise and fall points
    datflat = datr.remove_baseline(baseline=bl)
    tmp_peak = np.max(smooth(datflat.signal, n_smooth=.02, timebase=datflat.timebase)[1])
    (filt_tb, filtd) = smooth(datflat.signal, n_smooth=tfilt, timebase=datflat.timebase, causal=0)
    #  find a level where there is only two intersections
    for div in range(100, 1, -1):
        whigh = np.where(filtd > tmp_peak/div)[0]
        if (len(np.unique(np.diff(whigh))) == 1) and len(whigh)>len(filtd)//10:
            method = 'best'
            break
    else:
        method = 'simple'
        if debug>1:
            raise ValueError("can't find the pulse")
        else:
            bads.append(shot)
            #continue


    if method is 'simple':
        rf=dev.acq.getdata(shot,'H1_p_fwd_top')
        wrf=np.where(rf.signal > rf.signal.max()/2)[0]
        rftb = [min(rf.timebase[wrf]), max(rf.timebase[wrf])]
        rfmask = datflat.signal*0
        datwrf = np.where((datflat.timebase>rftb[0]) & (datflat.timebase<rftb[1]))[0]
        rfmask[datwrf] = 1
        dia = float(np.average(rfmask*datflat.signal))
    else:
        best.append(shot)
        wind = [filt_tb[whigh[0]]+tfilt/2, filt_tb[whigh[-1]]-tfilt/2]
        blfin = [wind[0]-dt, wind[0], wind[1], wind[1]+dt] 
        datflat_better = datr.remove_baseline(baseline=blfin)
        datint = datflat_better.integrate(baseline=None)
        dI = datint.reduce_time(blfin[2:]).signal.mean() - datint.reduce_time(blfin[:2]).signal.mean()
        dia = float(dI)/(blfin[2]-blfin[1])
        if (dbg > 0) and (len(best) < dbg):
            if len(best) == 1:
                # on the first hit, is there an ax1, ax2 pair that look right?
                try:               # if so use it for overlay.
                    ax1.mylabel, ax2
                    plt.sca(ax1)   # another check that the figure is not cleared
                except:
                    ax1 = plt.gca()
                    ax1.mylabel='get_diamag'
                    ax2 = ax1.twinx()
            plt.sca(ax1)
            datflat.plot_signals(color=None, labeleg='True', labelfmt='')
            ax1.plot(filt_tb, filtd)
            ax1.plot(plt.xlim(),[tmp_peak/div,tmp_peak/div],'--')
            plt.sca(ax2)
            datint.plot_signals(color='r')
            plt.sca(ax1)
    print('{s} {d:.3f}'.format(s=shot, d=dia))
    if h1db is not None:
        write_to_summary(shot, dict(dia_var=dia))

if len(best) == 0:
    print('no bests found')
else:
    ax1.legend()
if h1db is not None:
    h1db.commit()
