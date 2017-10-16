import numpy as np
import pyfusion
from pyfusion.debug_ import debug_
from matplotlib import pyplot as plt

def find_shot_times(shot = None, diag = 'W7X_LTDU_LP17_I', threshold=0.02, margin=[.2,.4], debug=0):
    """ return the actual interesting times in utc for a given shot, 
    based on the given diag 
    """
    dev_name = "W7X"
    #  diag = W7X_LTDU_LP20_I
    dev = pyfusion.getDevice(dev_name)
    nsold = pyfusion.NSAMPLES
    try:
        pyfusion.NSAMPLES = 2000
        dev.acq.repair = -1
        data = dev.acq.getdata(shot, diag, exceptions=())
    except Exception as reason:
        print('Exception ', str(reason))
        raise
    finally:
        pyfusion.NSAMPLES = nsold

    if not isinstance(data.timebase[1], int) and hasattr(data, 'params') and 'diff_dimraw' in data.params:
        data.timebase = data.params['diff_dimraw'].cumsum()

    tb = data.timebase
    wbig = np.where(np.abs(data.signal) > np.abs(threshold))[0]
    times = np.array([tb[wbig.min()] - margin[0]*1e9, tb[wbig.max()] + margin[1]*1e9], dtype=np.int)
    print('shot length={dt}, {times}'
          .format(dt=np.diff(times)/1e9, times=(times - tb[0])/1e9))
    if debug>0:
        data.plot_signals()
        plt.plot(times,[threshold, threshold],'o--r')
        plt.plot(times,[-threshold, -threshold],'o--r')
        plt.show()
    debug_(pyfusion.DEBUG, 3, key='find_shot_times')
    return(times)

if __name__ == "__main__":
    print(find_shot_times([20170913, 27], debug=1))
