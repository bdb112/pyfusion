import numpy as np
from time import time as seconds

def check_time_order(dd, allow_equal = True):
    shot = dd['shot']
    last_check = seconds()
    for s in np.unique(dd['shot']): 
        ws=np.where(s==shot)[0]
        if np.min(np.diff(dd['t_mid'][ws]))<0:
           print('time order problem in ' + str(s))
        else:
           print('OK'),
           if seconds() - last_check > 20:
               print('\n')
               last_check = seconds()


from pyfusion.data.DA_datamining import DA, report_mem
DA_test = DA('/data/datamining/old/PF2_130813_50_5X_1.5_5b_rms_1_diags.npz')
DA_test = DA('/data/datamining/DA300_384_rms1_b5a_f16_diags.npz')

dd = DA_test.da
dd=ddd

if 'phorig' not in dd.keys():
    print('*** phorig not present')
else:
    phorig = dd['phorig']
    ph0 = dd['phases'][:,0]
    b_0 = dd['b_0']
    if (ph0*phorig>=0).all():
        print('phases are untouched')
    else:
        pos  = ph0*phorig*b_0
        if (pos>=0).all():
            print('phase with B_0')
        elif (pos<=0).all():
            print('phase INVERSE with B_0')
        else:
            print('phase and B_0 mixed up')

check_time_order(dd)
