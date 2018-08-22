""" script to calculate effective resistance of shunts in the Langmuir probe array form DC tests
If two files are given, no residual calculation has no meaning.  If no open circuit test is given,
that of 1101 is used.  The lines below must be entered as a complete line.  Note the use of quotes 
when the string form of utc is used, to preserve spaces.  Not required if the shot number is given 
as integer64 utc values, separated by commas, and enclosed by [], with no spaces. 

# A test case using just a few probes, for speed, and the default OC test
run  pyfusion/examples/calibrate_probes.py ichans='W7X_UTDU_LPSOMEI' onek_shots="20171101 09:46:26,20171101 09:47:25"

Channel,      R_eff, conductance(S), resid1k/Imax, residOC/IMaxOC
W7X_UTDU_LP01_I,    2.206, -1.081e-03, None,   0.017
W7X_UTDU_LP11_I,    2.288, -1.122e-03, None,   0.026
W7X_UTDU_LP18_I,    2.225, -1.091e-03, None,   0.027

# a real example
run  pyfusion/examples/calibrate_probes.py ichans='W7X_UTDU_LPALLI' onek_shots="20171101 09:46:26,20171101 09:47:25,20171101 09:48:50" OC_shots="20171101 09:50:25,20171101 09:50:51,20171101 09:53:26"


"""
import pyfusion
import matplotlib.pyplot as plt
import numpy as np
from pyfusion.utils.time_utils import utc_ns

def get_fits(shots, dt=1):
    if plt.is_string_like(shots) and ',' in shots: 
        shots = [[utc_ns(nls), utc_ns(nls) + int(dt*1000000000)] for nls in shots.split(',')]

    i_data = [dev.acq.getdata(sh, ichans,exceptions=exceptions) for sh in shots]
    vsweep_data = [dev.acq.getdata(sh, vsweep) for sh in shots]
                   
    ichan_names = np.array([ch.name for ch in i_data[0].channels])
    gain_used = np.array([[i_dat.params[ch.name]['gain_used'] for ch in i_dat.channels] for i_dat in i_data])
    if gain_used.std(axis=0).sum() != 0:
        print('inconsistent gain_used')
    gain_used = gain_used[0]
    npts = len(vsweep_data[0].timebase)
    cm_part = np.array([np.sum(dat.signal, axis=1) for dat in i_data]).T/npts
    v_sum = np.array([float(np.sum(vdat.signal)) for vdat in vsweep_data])/npts

    fits = [np.polyfit(v_sum, cm, 1, full=True) for cm in cm_part]
    if len(fits[0][1]) == 0:
        res = [None for fit in fits]
    else:
        res = np.array([np.sqrt(fit[1][0]/len(shots))/np.max(np.abs(cm_part)) for fit in fits])

    return(dict(fits=[fit[0] for fit in fits], res=res,
                gain_used = gain_used, names=ichan_names))

_var_defaults = """
devname = 'W7X'
OC_shots = '20171101 09:50:25,20171101 09:50:51,20171101 09:53:26'
onek_shots = '20171101 09:46:26,20171101 09:47:25,20171101 09:48:50'
vsweep = 'W7X_KEPCO_U'
ichans = 'W7X_UTDU_LPBIGI'
delta_t = 1  # period of time analysed, starting from the given utc
exceptions=None  # set to [] to catch all exceptions
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if pyfusion.RAW != 0:
    pyfusion.RAW=0
    raise ValueError('pyfusion.RAW=1 makes no sense for this script - setting pyfusion.RAW to 0')
dev = pyfusion.getDevice(devname)
OC_fits = get_fits(shots=OC_shots, dt=delta_t)
onek_fits = get_fits(shots=onek_shots, dt=delta_t)
corr_conds = np.array([onek[0] - OC[0] for (onek, OC) in zip(onek_fits['fits'], OC_fits['fits'])])
R_nom = 1.0 / onek_fits['gain_used']
for i in range(len(R_nom)):
    if R_nom[i] == 4:
        R_nom[i] /= 2
    elif R_nom[i] in [110, 20, 11]:
        R_nom[i] /= 10
    else:
        raise ValueError('R_nom={R} not covered'.format(R=R_nom[i]))
  
print('\nChannel,      R_eff, conductance(S), resid1k/Imax, residOC/IMaxOC')
for (c, chan) in enumerate(onek_fits['names']):
    res1kc = format(onek_fits['res'][c], '7.2g') if onek_fits['res'][c] is not None else 'None'
    resOCc = format(OC_fits['res'][c], '7.2g') if OC_fits['res'][c] is not None else 'None'
    print('{nm:10s}, {Rseff:8.3f}, {cond:10.3e}, {res1kc}, {resOCc}'
          .format(nm=chan, Rseff=-1020*R_nom[c]*corr_conds[c], cond=corr_conds[c],
                  res1kc=res1kc, resOCc=resOCc))
