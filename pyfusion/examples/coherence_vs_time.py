"""
_PYFUSION_TEST_@@shot_number=92902 dev_name='H1Local'
"""
import pyfusion
import matplotlib.pyplot as plt
import numpy as np

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
diag1 = 'H1ISat'
diag2 = 'H1CameraTrig'
dev_name = "H1"
shot_number = 91700
hold=0
ylims=[-.1,1.1]
passband = [3e3,8e3]
stopband = [2e3,9e3]
period = [2e-4,1e-3]  #  interval (or list of intervals) over which coherence is   averaged (top hat)
plotit = 1
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
data1 = dev.acq.getdata(shot_number,diag1)
data2 = dev.acq.getdata(shot_number,diag2)
try:
    datap = dev.acq.getdata(shot_number,'H1Puff')
except:
    datap = None

fd1 = data1.filter_fourier_bandpass(passband=passband, stopband=stopband)
fd2 = data2.filter_fourier_bandpass(passband=passband, stopband=stopband)
t = fd1.timebase

#coh = runavg(fd1, period) * runavg(fd2, period)/(
#    np.sqrt(runavg(fd1, period)

if not np.iterable(period):
    period = [period]

if plotit:
    if hold == 0:
        plt.figure()
    data2.plot_signals(lw=.1, color='c')
for (i,per) in enumerate(period):
    t_coh, corr = runavg(t, fd1.signal * fd2.signal, per, return_time=1)
    coh = corr/np.sqrt(runavg(t, fd1.signal**2, per) * 
                       runavg(t, fd2.signal**2, per))
    if plotit:
        plt.plot(t_coh, -coh, color=['c','b'][i])  #  why minus??

if plotit:    

    plt.plot(plt.xlim(), [1,1], lw=0.3)
    plt.ylim(ylims)
    if datap is not None:
        axp = plt.twinx()
        axp.plot(datap.timebase, datap.signal, color='r')

    plt.show(0)
