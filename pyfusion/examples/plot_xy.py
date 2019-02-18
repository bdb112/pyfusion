""" Adapted from plot_signal.py - see bdbs_MDSplus_W7X/plot_lissa_smoothed.py for built-in test data.

run pyfusion/examples/plot_xy.py dev_name='W7X' shot_number=[180907,6] time_range=[0,0.01,.0000001] period=259.4 offs=-6 marker='-+'

run pyfusion/examples/plot_xy.py dev_name='W7X' shot_number=[20180911,24] diag_names=W7M_BRIDGE_V1,W7M_BRIDGE_APPROX_I
# at the moment time_range does not help for retrieved data
_PYFUSION_TEST_@@time_range=[4.6,4.6001]  
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
marker = '.r'
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

xdiag, ydiag = diag_name.split(',', 1)
if ',' in ydiag:
    ydiag, wdiag = ydiag.split(',')
else:
    wdiag = None
    
if time_range != []:
    if len(time_range) == 2 and 'APPROX' in ydiag:
        time_range += [2e-5 + 1e-7]  # assume downsampling if only two nums
    ROI = ' '.join([str('{t:.6f}'.format(t=t)) for t in time_range])
pyfusion.config.set('Acquisition:W7M','ROI', ROI)
dev = pyfusion.getDevice(dev_name)

dev.no_cache = True  # This allows direct access to server so we can use time context for efficiency
xdata = dev.acq.getdata(shot_number, xdiag, contin=not stop)
ydata = dev.acq.getdata(shot_number, ydiag, contin=not stop)
datlist = [xdata, ydata]
if wdiag is None:
    wdata = None
else:
    wdata = dev.acq.getdata(shot_number, wdiag, contin=not stop)
    datlist.append(wdata)

for dat in datlist:
    if dat is None:
        raise LookupError('data not found for {d} on shot {sh}'
                          .format(d=dat, sh=shot_number))
if len(time_range) == 2:  
    for dat in datalist:
        dat.reduce_time(time_range, copy=False)
#   xdata = xdata.reduce_time(time_range)
#   ydata = ydata.reduce_time(time_range)


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

"""  Not needed now I have implemented valid
if 'APPROX_I' in ydiag and tuple(shot_number) > (20180912,15) and dev_name == 'W7M' and :
    print('================ Applying gain factor 10 ===============')
    yarr = yarr * 10.
"""

if len(xdata.signal) < period + np.abs(offs):
    raise LookupError('Not enough samples to average {l}'.format(l=len(xdata.signal)))
fig, axs = plt.subplots(1, 2)
w = np.where(wdata.signal > 0.1) if wdata is not None else range(len(xdata.timebase))
axs[0].plot(xarr[w], yarr[w], ',', **plotkws)
bckwargs = dict(maxnum=maxcyc, period=period)
xbc, numused = boxcar(sig=xarr, return_numused=True, **bckwargs)
ybc = boxcar(sig=yarr, **bckwargs)

print('Now overlay one cycle')
xbcreal = boxcar(sig=xarr, period=period, maxnum=1)
ybcreal = boxcar(sig=yarr, period=period, maxnum=1)

axs[1].plot(xbcreal, ybcreal, '.', label='real time, no delay', markersize=4, **plotkws)
axs[1].plot(xbc, ybc, '.', label='{nc} cyc., no delay'.format(nc=numused), markersize=4, **plotkws)
axs[1].plot(rotate(xbc, offs), ybc, marker, label='{nc} cyc., vdel {ns}ns'
            .format(nc = numused, ns=offs*-100), **plotkws)
axs[0].set_xlabel('V')
axs[0].set_ylabel('A')  #  avoids interference
if len(time_range) > 0:
    tr ='[{0}]'.format(', '.join([str(np.round(t,8))
                                  for t in [xdata.timebase[0], xdata.timebase[-1]]]))
else:
    tr = ROI
tr += str(' per = {p:.2f}'.format(p=period))    
desc = 'test' if tuple(shot_number) < tuple([990000]) else 'program'
fig.suptitle(ydiag + ' ' + desc + ': ' + str(shot_number) + ': ' + tr,
             size='large')

"""
useful trick to get_suptitle()
gcf().get_children()[-1].get_text()


data.plot_signals(suptitle='shot {shot}: '+diag_name, t0=t0, sharey=sharey, downsamplefactor=max(1, decimate),
                  fun=fun, fun2=fun2, labeleg=labeleg, **plotkws)
"""
# for shot in range(76620,76870,10): dev.acq.getdata(shot,diag_name).plot_signals()
if labeleg:
    axs[1].legend(loc='best')
fig.show()
