""" Used to examine the spectra of all or selected channels used in the Mirnov
calibration, to detect tiny signals.  This helps in identifying dead
channels, and making sure that exciter probe is positioned so that data is good enough
"""
import pyfusion
import matplotlib.pyplot as plt
import numpy as np

_var_defaults = """
dev_name = "H1Local"
diag_name = "H1Poloidal1"
shot_number = 89089
freq=30e3
delta=500
sharey=False
sel=None        # specify indices of the channel to plot
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)

for i,(sig,chan) in enumerate(zip(data.signal,data.channels)):
    if sel is None or i in sel:
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(data.timebase),np.average(np.diff(data.timebase)))),
                 np.fft.fftshift(np.abs(np.fft.ifft(sig-np.average(sig)))),
                 label=str(i)+': '+chan.config_name, linestyle=['-','--','-.',':'][i//7])

plt.title('{s}: {d}'.format(s=shot_number, d=diag_name))
plt.yscale('log')
plt.ylim(1e-5, 1e1)
plt.xlim(freq-delta, freq+delta)
plt.legend(prop={'size':'x-small'})
plt.show(block=0)
