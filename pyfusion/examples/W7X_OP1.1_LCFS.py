from lukas.PickledAssistant import lookupPosition
import numpy as np
#_PYFUSION_TEST_@@SKIP
for lim in ['lower','upper']:
    print('\n ===== Limiter {l} ===== '.format(l=lim))
    print('Probe  X,       Y,      Z,     dLCFS')
    for LPnum in range(1,20+1):
        X,Y,Z,dLC = lookupPosition(LPnum, lim)
        print('LP{LPnum:02}: {xyz}: {dLC:6.2f}'
              .format(LPnum=LPnum, xyz=np.round([X,Y,Z], 2), dLC=dLC))
