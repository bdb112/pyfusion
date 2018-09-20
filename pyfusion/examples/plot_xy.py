""" from plot_signal.py 
run pyfusion/examples/plot_xy.py dev_name='W7X' shot_number=[20180911,24] diag_names=W7M_BRIDGE_V1,W7M_BRIDGE_APPROX_I
"""
import pyfusion
from pyfusion.utils import boxcar, rotate
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt
import numpy as np

_var_defaults = """
ROI = '.9 1 1e-5'
dev_name = "W7M"
diag_name = "W7M_BRIDGE_V1,W7M_BRIDGE_APPROX_I"
shot_number = [20180911,24]
decimate=1   #  1 is no decimation, 10 is by 10x
plotkws={}
hold=0  # 0 means erase, 1 means hold
labeleg='False'
time_range=[]
t0=0
stop=True  # if False, push on with missing data  - only makes sense for multi channel
offs = -4
period = 200
maxcyc = 100
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

def notimp():
    raise NotImplemented('yet: sorry!')

# save the start utc if there is already a 'data' object (***need to run -i to keep)
if 'W7' in dev_name:
    if 'utc0' in locals() and utc0 is not None:
        print('using previous utc')
    else:
        if 'data' in locals() and hold !=0:
            utc0 = data.utc[0] 

if time_range != []:
    if len(time_range) == 2:
        time_range += [2e-5 + 1e-7]  # assume downsampling
    ROI = ' '.join([str('{t:.6f}'.format(t=t)) for t in time_range])
pyfusion.config.set('Acquisition:W7M','ROI', ROI)
dev = pyfusion.getDevice(dev_name)
xdiag, ydiag = diag_name.split(',')

dev.no_cache = True
xdata = dev.acq.getdata(shot_number, xdiag, contin=not stop)
ydata = dev.acq.getdata(shot_number, ydiag, contin=not stop)
for dat in [xdata, ydata]:
    if dat is None:
        raise LookupError('data not found for {d} on shot {sh}'
                          .format(d=dat, sh=shot_number))
"""if len(t_range) > 0:  
    notimp()
    data = data.reduce_time(time_range)
"""

if hold == 0: plt.figure()

if 'W7X' in dev_name:
    if t0 is None:
        pgms = pyfusion.acquisition.W7X.get_shot_list.get_programs(shot=shot_number)
        pgm = pgms['{d}.{s:03d}'.format(d=shot_number[0], s=shot_number[1])]
        utc0 = pgm['trigger']['5'][0]
        # maybe should be in data/plots.py, but config_name not fully implemented
    if 'utc0' in locals() and utc0 is not None:
        t0 = (utc0 - data.utc[0])/1e9

xarr = xdata.signal
yarr = -ydata.signal
fig, axs = plt.subplots(1, 2)
axs[0].plot(xarr, yarr, ',', **plotkws)

bckwargs = dict(maxnum=maxcyc, period=period)
xbc = boxcar(sig=xarr, **bckwargs)
ybc = boxcar(sig=yarr, **bckwargs)

xbcreal = boxcar(sig=xarr, period=period, maxnum=1)
ybcreal = boxcar(sig=yarr, period=period, maxnum=1)

axs[1].plot(xbcreal, ybcreal, '.', label='real time, no delay', markersize=4, **plotkws)
axs[1].plot(xbc, ybc, '.', label='{nc} cyc., no delay'.format(nc=maxcyc), markersize=4, **plotkws)
axs[1].plot(rotate(xbc, offs), ybc, '.r', label='{nc} cyc., vdel {ns}ns'
            .format(nc = maxcyc, ns=offs*-100), **plotkws)
axs[0].set_xlabel('V')
axs[0].set_ylabel('A')  #  avoids interference
if len(time_range) > 0:
    tr = str(np.round([time_range[0], time_range[1]], 8))
else:
    tr = ROI
# desc = 'test' if tuple(shot) < 1e8 else 'program'
desc = 'program'
fig.suptitle(desc + ': ' + str(shot_number) + ': ' + tr, size='x-large')


"""
data.plot_signals(suptitle='shot {shot}: '+diag_name, t0=t0, sharey=sharey, downsamplefactor=max(1, decimate),
                  fun=fun, fun2=fun2, labeleg=labeleg, **plotkws)
"""
# for shot in range(76620,76870,10): dev.acq.getdata(shot,diag_name).plot_signals()
if labeleg:
    axs[1].legend(loc='best')
fig.show()
