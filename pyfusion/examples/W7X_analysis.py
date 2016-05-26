# plotting of the pickle files produce by process_swept_Langmuir.py

import pyfusion
from pyfusion.acquisition.W7X.get_shot_info import get_shot_info
import matplotlib.pyplot as plt
import numpy as np

_var_defaults = """
pfile='LP_20160310_9_W7X_L57_LP1_I_5120_20160311.pickle'
pfile='LP_20160310_9_W7X_L57_LP1_I_5120_20160321.pickle'
echdiag = 'W7X_TotECH'
dev_name = 'W7X'
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

import pickle
#run -i pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name=W7X_TotECH shot_number=[20160310,9] hold=0 sharey=2
plt.figure()
plt.rcParams['lines.linewidth']=2
q=1.e-19
mi=1.67e-27
A=1.8e-6
shot_number = [int (s) for s in pfile.split('_')[1:3]]
results=pickle.load(open(pfile))
if isinstance(results, dict):
    tresults = results['tresults']
    params = results['params']
else:
    tresults = results

ax = plt.subplot(1,1,1)
ax.plot(array(tresults).T[0],1/(0.6*q*A)*np.sqrt(mi/(q))/1e18*array(tresults).T[3]/np.sqrt(array(tresults).T[1]),label=r'~$n_e^{18}$')
plt.ylim(0,8)
ax2=ax.twinx()
prb = '_'.join(pfile.split('W7X_')[1].split('_')[0:2])
ax2.plot(array(tresults).T[0],array(tresults).T[1],'r',label=r'$T_e$ '+prb)

dev = pyfusion.getDevice(dev_name)
echdata = dev.acq.getdata(shot_number,echdiag)

ax2.plot(echdata.timebase-4,echdata.signal/1000,'g',label='totECH (MW)')
plt.ylim(0,80)
plt.xlim(0.5,2.2)
ax2.legend()
ax.legend(loc='upper left')
plt.title('{c} {s}'.format(c=get_shot_info(*shot_number).strip(),s=shot_number))
plt.show()


