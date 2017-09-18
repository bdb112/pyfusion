from __future__ import print_function
       
import os
from get_url_parms import get_signal_url
from get_shot_info import get_shot_utc
# works nicely for op1 and 1.2a for Gyro, get error on planar coils for 20160310,23
# Name for channelnr(12) is: ''Current of AAE14 through planar coils type A'' Not ''Current planar coils type A'' !




for path in ['CBG_ECRH/A1/medium_resolution/Rf_A1','AAE_MainCoils/Currents/Current planar coils type A']:
    for shot in [[20160310,23], [20170913,24]]:
        utc = get_shot_utc(shot)
        # this is the usual middle step, but
        # print(get_signal()+'scaled/_signal.json?filterstart={f}&filterstop={t}'.format(f=utc[0], t=utc[1]))
        # this one works directly
        url = os.path.join(get_signal_url(path),'scaled/_signal.json?from={f}&upto={t}'.format(f=utc[0], t=utc[1]))
        print(url,'\n--------------')
        retcode = os.system('wget -q -O - "{url}"|head -c 1000|fold -s -w80'.format(url=url))
        print('\nreturns ', retcode, '---------')
