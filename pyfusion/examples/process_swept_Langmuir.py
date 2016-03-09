""" process sinusoidally swept Langmuir probes (W7X) 

# good compensation
# No - wrong gains run pyfusion/examples/process_swept_Langmiur.py diag_name=W7X_L57_LP7_I "atten=(1.015e-4,-2e-5)"

Best so far 10 Ohm!!
run pyfusion/examples/process_swept_Langmuir.py diag_name=W7X_L57_LP7_I "atten=(3.4e-4,-6.5e-5)" maxits=1000 t_range=[1,1.1] debug=3

Best so far! 1Ohm!!
run pyfusion/examples/process_swept_Langmuir.py diag_name=W7X_L57_LP10_I "atten=(.5e-5,-5e-5)" maxits=1000 t_range=[1,1.1] debug=3
"""

import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin
import pyfusion.conf as conf
from scipy.fftpack import hilbert
from pyfusion.data import amoeba

def LPchar(v, Te, Vp, I0):
    return(I0 * (1 - exp((v-Vp)/Te)))

def error_fun(var, data):
    """ return RMS error """
    Te, Vp, I0 = var
    v = data['v']
    i = data['i']
    varr = np.linspace(np.min(v), np.max(v))
    plt.plot(varr,  LPchar(varr, Te, Vp, I0), 'g', linewidth=2, 
             alpha=min(1,1./np.sqrt(maxits)))
    err = np.sqrt(np.mean((i - LPchar(v, Te, Vp, I0))**2))
    if debug>1: print(Te, Vp, I0, err)
    return(-err)  #  amoeba is a maximiser

def LPfit(v,i,init=dict(Te=50, Vp=15, I0=None)):
    Te = init['Te']
    Vp = init['Vp']
    I0 = init['I0']
    if I0 is None:
        I0 = np.average(np.clip(i,0,1000))

    var = [Te, Vp, I0]
    scale = np.array([Te, Te/3, I0])
    fit_results = amoeba.amoeba(var, scale, error_fun, itmax = maxits, 
                                data = dict(v=v, i=i))
    Te, Vp, I0 = fit_results[0]
    residual, mits = fit_results[1:3]
    plt.title('Te = {Te:.1f}eV, Vp = {Vp:.1f}, Isat = {I0:.3g}, resid = {r:.2e} {mits} its '
              .format(Te=Te, Vp=Vp, I0=I0, r=-residual, mits=mits))
    return(fit_results)


_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_L57_LP3_I"
shot_number = [20160308,25]
sharey=2
plotkws={}
atten = [1e-4,1e-5]
fun=myiden
dphi = 0.0  # shift between measured current and voltage - only for testing
fun2=myiden2
hold=0
debug=1
t_range=[1.34,1.36]
maxits=1000
sweep_default="W7X_L57_LP1_U"
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)

if hold==0: plt.figure()
data.plot_signals(suptitle='shot {shot}: '+diag_name, sharey=sharey,
                  fun=fun, fun2=fun2, **plotkws)
imeas = data.signal
params = conf.utils.get_config_as_dict('Diagnostic', diag_name)
if 'sweep' in params:
    sweep_diag = params['sweep']
else:
    if debug>0:
        sweep_diag = sweep_default
        print('defaulting sweep_diag to {dsd}'.format(dsd=sweep_diag))

sweep_data =  dev.acq.getdata(shot_number, sweep_diag)
dt = sweep_data.utc[0] - data.utc[0]
if dt != 0:
    print('******* Warning - timebases differ by {dt:.3f} us'
          .format(dt=dt/1e3))



sweepVmeas = sweep_data.signal
usable_length = min(len(sweepVmeas),len(imeas)) 
sweepVmeas = sweepVmeas[0:usable_length]
imeas = imeas[0:usable_length]
tb = data.timebase[0:usable_length]

# first need to shift IMeas relative to sweep.  Can use hilbert to 
# shift the sweep instead.
sweepQ = hilbert(sweepVmeas)
sweepV = cos(dphi) * sweepVmeas - sin(dphi) * sweepQ


if debug>0:
    plt.plot(tb, sweepV*atten[0],'r')
#plt.plot(sweep_data.timebase, sweep_data.signal/atten,'r')

plt.plot(tb, imeas - sweepV*atten[0],'c', label='corrected for R')
iprobe = imeas - sweepV*atten[0] - hilbert(sweepV)*atten[1]
plt.plot(tb, iprobe,'g',label='R and phase')
plt.legend(prop=dict(size='small'))
plt.figure()

tb = sweep_data.timebase
inds = np.where((tb>t_range[0]) & (tb<t_range[1]))[0] 
plt.plot(sweepV[inds],iprobe[inds],'r.',alpha=min(1,4/np.sqrt(len(inds))))

results = LPfit(sweepV[inds],iprobe[inds])

plt.figure()
plt.plot(tb[inds],sweepV[inds]*np.max(np.abs(atten)))
plt.plot(tb[inds],iprobe[inds])
plt.show(block=0)
