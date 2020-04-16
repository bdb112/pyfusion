from pyfusion.acquisition.W7X.lukas_approx_SOL.PickledAssistant import lookupPosition
import numpy as np
#_PYFUSION_TEST_@@SKIP
allLP = []
for lim in ['lower','upper']:
    print('\n ===== Limiter {l} ===== '.format(l=lim))
    print('Probe  X,       Y,      Z,     dLCFS')
    for LPnum in range(1,20+1):
        X,Y,Z,dLC = lookupPosition(LPnum, lim)
        allLP.append([lim, LPnum, [X/1e3, Y/1e3, Z/1e3], dLC])
        print('LP{LPnum:02}: {xyz}: {dLC:6.2f}'
              .format(LPnum=LPnum, xyz=np.round([X,Y,Z], 2), dLC=dLC))
