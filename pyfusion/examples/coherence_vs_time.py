"""

W7X Note - for use on probe current signals, the sweep voltage pickup can dominate - ideally should be removed, although voltage being more sinusoidal is a better candidate and should be used for this test if possible

_PYFUSION_TEST_@@shot_number=92902 dev_name='H1Local'
"""
import pyfusion
import matplotlib.pyplot as plt
import numpy as np
import scipy

"""
def runavg(tseries, period=1e-3):
    csum = np.cumsum(tseries.signal)
    nsamp = int(round(period / np.mean(np.diff(tseries.timebase))))
    return(tseries.timebase[nsamp//2:-nsamp//2], csum[0:nsamp] - csum[nsamp:])
"""


def runavg(t, y, period=1e-3, return_time=False):
    csum = np.cumsum(y)
    nsamp = int(round(period / np.mean(np.diff(t))))
    if return_time:
        return(t[nsamp//2:-nsamp//2], csum[0:-nsamp] - csum[nsamp:])
    else:
        return(csum[0:-nsamp] - csum[nsamp:])

_var_defaults = """
diag1 = 'H1_ISat'
diag2 = 'H1_CameraTrig'
diag_extra = "H1_Puff"  # Use None to suppress
dev_name = "H1"
shot_number = 91700
hold=0
ylims=[-1,1.1]
passband = [3e3,8e3]
stopband = [2e3,9e3]
corrdelay = 1
period = [2e-4,1e-3]  #  interval (or list of intervals) over which coherence is averaged (top hat)
plotit = 1
t_range=[]
AC=1  # if True, subtract the average first
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if shot_number in  [0,1]:
    import MDSplus as MDS
    tr = MDS.Tree('h1data', 0)
    # get the current (0) or the next shot (1)
    shot_number = tr.getCurrent('h1data') + shot_number

dev = pyfusion.getDevice(dev_name)
data1 = dev.acq.getdata(shot_number, diag1)
data2 = dev.acq.getdata(shot_number, diag2)

if len(t_range) != 0:
    data1.reduce_time(t_range, copy=0)
    data2.reduce_time(t_range, copy=0)

comstart = max(data1.timebase[0], data2.timebase[0])
comend = min(data1.timebase[-1], data2.timebase[-1])
comend -= 0.05*(comend - comstart) # allow room to take more samples in reduce time
# if the timebases are not identical, this doesn't always work, so take off a half interval to avoid representational errs
data1 = data1.reduce_time([comstart, comend - np.diff(data1.timebase)[0]/2],fftopt=1)
data2 = data2.reduce_time([comstart, comend - np.diff(data2.timebase)[0]/2],fftopt=1)

try:
    datap = dev.acq.getdata(shot_number, diag_extra)
except:
    datap = None

if len(passband) != 0:
    fd1 = data1.filter_fourier_bandpass(passband=passband, stopband=stopband)
    fd2 = data2.filter_fourier_bandpass(passband=passband, stopband=stopband)
    dt_dodgy = 0.5/stopband[0]
    bpstr=',bp {frf:.1f}-{tof:.1f}kHz'.format(frf=passband[0]/1e3, tof=passband[1]/1e3)
else:
    fd1 = data1
    fd2 = data2
    dt_dodgy = None

t = fd1.timebase


#coh = runavg(fd1, period) * runavg(fd2, period)/(
#    np.sqrt(runavg(fd1, period)

if not np.iterable(period):
    period = [period]

y1 = fd1.signal
y2 = fd2.signal

if AC:
    y1 = y1 - np.average(y1)
    y2 = y2 - np.average(y2)

if plotit:
    if hold == 0:
        plt.figure()
    data2.plot_signals(lw=.1, color='c', label=diag2)
ax = plt.gca()

for (i,per) in enumerate(period):
    t_coh, corr = runavg(t, y1 * y2 , per, return_time=1)
    coh = corr/np.sqrt(runavg(t, y1**2, per) * 
                       runavg(t, y2**2, per))
    if plotit:
        ax.plot(t_coh, -coh, color=['c','b'][i], label='avg. over {per}ms'.format(per=per*1e3))  #  why minus

if corrdelay:
    fd1 = scipy.fft(data1.signal)
    fd2 = scipy.fft(data2.signal)
    corr = scipy.ifft(fd1 * scipy.conj(fd2))
    imax = np.argmax(np.abs(corr))
    if imax > len(corr)/2:
        imax = imax - len(corr)
    print('time delay = {td:.4g}'.format(td=imax*np.diff(data1.timebase)[0]))

if plotit:    

    ax.plot(plt.xlim(), [1,1], lw=0.3)
    ax.set_ylim(ylims)
    if datap is not None:
        axp = plt.twinx()
        axp.plot(datap.timebase, datap.signal, color='r')
    from matplotlib.patches import Rectangle
    someX, someY = 23.5, -0.5
    xl = [data1.timebase.min(), data1.timebase.max()]
    yl = ax.get_ylim()
    #  try to indicate region spoilt by filtering etc
    dodgy_total =  dt_dodgy + np.max(period)/2
    ax.add_patch(Rectangle((xl[0], yl[0]), dodgy_total, yl[1]-yl[0], facecolor='lightgray',
                           edgecolor='lightgray', label='dubious interval'))
    ax.add_patch(Rectangle((xl[1] - dodgy_total, yl[0]), dodgy_total, yl[1]-yl[0], facecolor='lightgray',
                           edgecolor='lightgray'))

    plt.legend(loc='best', fontsize='medium')
    plt.title('{d1} <coh> {d2} {bp}'.format(s=str(shot_number), d1=diag1, d2=diag2, bp=bpstr))
    plt.show(0)
