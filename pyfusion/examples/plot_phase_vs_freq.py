from matplotlib import pyplot as plt
import numpy as np
import pyfusion
import pyfusion.utils
from scipy import fftpack as fft
from pyfusion.utils.utils import fix2pi_skips, modtwopi


_var_defaults = """ """
diag_name = ''
debug = 0 # 
dev_name = 'W7X'   # 'LHD'
exception = Exception  # to catch all exceptions and move on
time_range = [5.8, 5.81]
dt = 1e-3
help = 0
verbose = 0
shot_number = None
stop = False  # True will stop to allow debugging
fsel = []
# sharex = 'col'
""" """
exec(_var_defaults)
exec(pyfusion.utils.process_cmd_line_args())
if help == 1:
    print(__doc__)
    exit()
fig, ((axa, axspare), (axp, axparray)) = plt.subplots(2, 2, sharex='col')
#axp = plt.gca()
# time_range = [5.65,5.75]
device = pyfusion.getDevice(dev_name)
d = device.acq.getdata(shot_number, diag_name, time_range=time_range)
seg = d.reduce_time([time_range[0], time_range[0] + dt])
freqs = fft.fftfreq(len(seg.timebase), np.average(np.diff(seg.timebase)))
fsel = arange(0, len(freqs)//2) if fsel == [] else fsel
freqs = freqs[0:len(freqs)//2] # ditch neg f

FTs = []  # save the FT and the channel index for later processing.
isigs = []
for (isig, sig) in enumerate(seg.signal):
    try:
        FT = np.fft.fft(sig)
        FTs.append(FT)
        isigs.append(isig)
    except:
        continue
    ls = ['-', '--', ':', '-.'][(isig // 7) % 4]
    label=seg.channels[isig].name.split('_',1)[-1]
    axp.plot(freqs[fsel]/1e3, np.angle(FT[fsel]), ls=ls, label=label) 
    # axp.plot(freqs[fsel]/1e3, 2*np.pi + np.angle(FT[fsel]), color=axp.get_lines()[-1].get_color())
    axa.plot(freqs[fsel]/1e3, np.abs(FT[fsel]), ls=ls, label=label, color=axp.get_lines()[-1].get_color())
axa.legend(fontsize='xx-small', loc='best',ncol=1 + len(isigs)//8)  # put legend on amplitude plot as it has more room
fig.suptitle('Shot {sh}: {di}, <t>={tavg:.4g}, dt={dt}'
             .format(sh=str(shot_number), di=diag_name, dt=dt, tavg=np.average(seg.timebase)))
axparray.set_xlabel('<t>={tavg:.4g} dt={dt}'
                    .format(sh=str(shot_number), di=diag_name, dt=dt, tavg=np.average(seg.timebase)))

plt.show()


def plot_array_phase(freq, ax=plt.gca(), fixp=False, ref_probe=0, ref_offset=0):
    """  Warning - this really should have FS passed as an arg
    Also, to fix 2Pi skips - if there are a number of noisy graphs that 
    should be similar.  Chose the 2pi skip that keeps a graph closest to 
    an already processed one (or the ensemble average?) (how??)

    """

    if not isinstance(freq, int):
        findex = np.argsort(np.abs(freqs - freq))[0]
        print('freq {freq} index {findex}'.format(**locals())),
    else:
        findex = freq

    phaselist_orig = [np.angle(FTs[isig][findex]) for isig in isigs]
    phaselist = np.copy(phaselist_orig)  # save the original for debugging

    phaselist = fix2pi_skips(phaselist) if fixp else phaselist
    phaselist = phaselist - phaselist[ref_probe] if ref_probe > -999 else phaselist
    if (ref_probe) > -999 and (not fixp):  #  use mod2pi to
        phaselist = modtwopi(phaselist, offset=ref_offset)
    ax.plot(phaselist, label=str('{f:.1f}kHz'.format(f=freqs[findex]/1e3)))
    ax.legend(fontsize='xx-small', loc='best',ncol=1 + min(3, len(isigs)//6))
    plt.show()
