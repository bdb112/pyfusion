import numpy as np
import pyfusion

def find_shot_times(shot = None, diag = 'W7X_LTDU_LP17_I', threshold=0.02, margin=[.2,.2]):
    """ return the actual interesting times for a given shot, based on the given diag 
    """
    dev_name = "W7X"
    #  diag = W7X_LTDU_LP20_I
    dev = pyfusion.getDevice(dev_name)
    nsold = pyfusion.NSAMPLES
    try:
        pyfusion.NSAMPLES = 2000
        dev.acq.repair = -1
        data = dev.acq.getdata(shot, diag)
    except Exception as reason:
        print('exception ', str(reason))
    finally:
        pyfusion.NSAMPLES = nsold

    tb = data.timebase
    wbig = np.where(np.abs(data.signal) > threshold)[0]
    times = np.array([tb[wbig.min()] - 1e9*margin[0], tb[wbig.max()] + 1e9*margin[1]], dtype=np.int)
    print('shot length {dt}, {times}'.format(dt = np.diff(times)/1e9, times=times))
    return(times)

if __name__ == "__main__":
    print(find_shot_times([20170913, 27]))
