""" quick FFT, to check hum frequency 
goes with plot_text_fft.py, used to deduce clk divisor by looking at hum"""
import pyfusion, sys
import pylab as pl
import numpy as np 

_var_defaults="""

shot_range=[27233]
plot=True
exception=Exception
diag_name='HMP01'
dev_name="LHD"
"""

exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if pl.is_string_like(shot_range):
    print('loading from file %s ' %(shot_range))
    shot_range = np.loadtxt(shot_range)

device = pyfusion.getDevice(dev_name)

for shot in shot_range:
    try:
        data = device.acq.getdata(shot, diag_name)
        #data = pyfusion.load_channel(shot,chan_name)
        sig=data.signal
        sigac=sig-np.average(sig*np.blackman(len(sig)))
        fs = abs(np.fft.fft(sigac*np.blackman(len(sig))))
        if plot:
            pl.semilogy(fs[0:2000],hold=0)
            pl.title('shot %d, %s' % (shot, diag_name))
            if not(pl.isinteractive()): pl.show()


        nsamp=len(sig)
    # 30 and 45 Hz signals are meant to be quiet
        freqs=(0.5 + array([30, 45, 60,120,180,240,360,720])*nsamp/1e6)
        ff = array([int(f) for f in freqs])

        print (" %d %s " % (shot, ' '.join([str('%.1f' % fs[int(f)]) for f in ff])))
    except exception:
        pass
    except:
        print('shot %d not processed: %s' % 
                          (shot,sys.exc_info()[1].__repr__()))
