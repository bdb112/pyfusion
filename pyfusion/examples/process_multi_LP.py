""" Try to reduce noise especially on bridge probe by staggering analysis and emphasising the
result with the best fit I think

e.g run pyfusion/examples/process_multi_LP.py  tstart=1.00182 tend=1.00186 nfull=5 nhalf=0 "extra_kwargs=dict(fit_params=dict(cycavg=[200,100,-4]),plot=3,suffix='full82_400')" plot_DA=98 program=[20181010,31]
"""

import numpy as np
from matplotlib import pyplot as plt
from pyfusion.utils.process_cmd_line_args import process_cmd_line_args
from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup
#from copy import deepcopy
from pyfusion.data.DA_datamining import Masked_DA, DA

kwargs = dict(t_comp=[0, 0], threshchan=-1, initial_TeVfI0=dict(Te=20, Vf=1, I0=None), fit_params=dict(alg='leastsq', maxits=300, esterr='trace', track_ratio=1.2, cycavg=[200, 1, -3]), debug=0, return_data=1)

# the ones we will change most
kwargs.update(dict(filename='/tmp/pmulti_{s0}_{s1}_{t_start}_{dtseg}'))

_var_defaults = """
extra_kwargs = {}
program = [20181010, 31]
program = [20180927, 30]  # short for tests
program = [20180920, 29]  # clean, low te
seg_len = 200
tstart = 1.0
tend = 1.001
toffs = None
plot_DA = 21
nfull = 5 # the number of rotations of one full sample set
nhalf = 2 # the number of rotations of a half sample set - only 2 makes sense for one period
"""
exec(_var_defaults)
exec(process_cmd_line_args())

# pull out any fit params, and update specifically so we don't trash the original
extra_fit_params = extra_kwargs.pop('fit_params')
kwargs['fit_params'].update(extra_fit_params)
kwargs.update(extra_kwargs) # then can update the rest


LP = Langmuir_data(program, 'W7M_BRIDGE_ALLI', 'W7M_BRIDGE_V1', dev_name='W7M')

dt = np.average(np.diff(LP.imeasfull.timebase))

plt.figure()
# extract 10 cycles at t=1
vmeas_t0 = LP.vmeasfull.reduce_time([tstart, tstart + 10 * seg_len * dt], copy=True)
if len(vmeas_t0.timebase) < seg_len:
    raise LookupError('time window not contained in data segment?')

plt.plot(vmeas_t0.signal[0:seg_len])
if toffs is None:
    toffs = -seg_len * np.angle(np.fft.fft(vmeas_t0.signal[0:seg_len])[1]) / (2 * np.pi) * dt

print(tstart)
tstart = tstart + toffs
print(tstart)

alldas = []
for ts in np.linspace(tstart, tstart + seg_len * dt, nfull, endpoint=False):
    results = LP.process_swept_Langmuir(t_range=[ts, tend], dtseg=seg_len, plot_DA=plot_DA, **kwargs)
    if hasattr(LP, 'da'):
        alldas.append(LP.da)

# this one must start at a V maximum
half_kwargs = dict(dtseg=seg_len // 2, **kwargs)
fit_params = half_kwargs['fit_params']

cycavg_h = fit_params['cycavg']
if cycavg_h is not None and (seg_len // 2) < cycavg_h[0]:
    cycavg_h = [seg_len // 2, 1, cycavg_h[2]]
    fit_params.update(dict(cycavg=cycavg_h))
    half_kwargs.update(dict(fit_params=fit_params))
    
for ts in np.linspace(tstart, tstart + seg_len * dt, nhalf, endpoint=False):
    results = LP.process_swept_Langmuir(t_range=(ts, tend), plot_DA=plot_DA, **half_kwargs)
    if hasattr(LP, 'da'):
        alldas.append(LP.da)

bigda = DA(alldas[0].copyda())
for da in alldas[1:]:
    bigda.append(da)

            
