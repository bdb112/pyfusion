from __future__ import print_function

import numpy as np
import pyfusion
from pyfusion.debug_ import debug_
from matplotlib import pyplot as plt

def find_shot_times(shot = None, diag = 'W7X_UTDU_LP10_I', threshold=0.4, margin=[.3,.4], debug=0, exceptions=(LookupError)):
    """ return the actual interesting times in utc for a given shot, 
    based on the given diag.  Use raw data to allow for both 1 and 10 ohm resistors (set above common mode sig)
    """
    dev_name = "W7X"
    #  diag = W7X_UTDU_LP10_I  # has less pickup than other big channels
    dev = pyfusion.getDevice(dev_name)
    nsold = pyfusion.NSAMPLES
    pyfusion.RAW = 1  # allow for both 10 and 1 ohm sensing resistors
    try:
        pyfusion.NSAMPLES = 2000
        dev.acq.repair = -1
        data = dev.acq.getdata(shot, diag, exceptions=())
    except exceptions as reason:
        print('Exception suppressed: ', str(reason))
        return None
    except Exception as reason:
        print('Exception NOT suppressed: ', str(reason))
        raise
        
    finally:
        # this is executed always, even if the except code returns
        pyfusion.NSAMPLES = nsold
        pyfusion.RAW = 0
        print('params restored')
        debug_(pyfusion.DEBUG, 3, key='find_shot_times')
                    
    if not isinstance(data.timebase[1], int) and hasattr(data, 'params') and 'diff_dimraw' in data.params:
        data.timebase = data.params['diff_dimraw'].cumsum()

    tb = data.timebase
    wbig = np.where(np.abs(data.signal) > np.abs(threshold))[0]
    if len(wbig) < 5:
        pyfusion.utils.warn('Too few points above threshold on shot {shot}'.format(shot=str(shot)))
        return None
    times = np.array([tb[wbig.min()] - margin[0]*1e9, tb[wbig.max()] + margin[1]*1e9], dtype=np.int64)
    print('shot length={dt}, {times}'
          .format(dt=np.diff(times)/1e9, times=(times - tb[0])/1e9))
    if debug>0:
        data.plot_signals()
        plt.plot(times,[threshold, threshold],'o--r')
        plt.plot(times,[-threshold, -threshold],'o--r')
        plt.show()
    return(times)

if __name__ == "__main__":
    print(find_shot_times([20170913, 27], debug=1))
