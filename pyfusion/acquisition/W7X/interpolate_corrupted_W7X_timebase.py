import numpy as np
import pyfusion
from pyfusion.debug_ import debug_

def fake_regenerate_dim(x):
    return(x['timebase'])

# following is probably only for debugging?
from pyfusion.acquisition.W7X.get_shot_info  import get_shot_utc
from matplotlib import pyplot as plt

def interpolate_corrupted_W7X_timebase(signal_dict):
    """ Early saved data incorrectly regenerated dim to seconds, resolution
    was only ~ 50ms, due to use of uint8 I think, which works if the timebase is perfect, but once the steps are greater than 255*2000 us->, 
    """
    # trick to avoid circular import error - regenerate_dim should be standalone
    from pyfusion.acquisition.W7X.fetch import regenerate_dim
    print('==============  Try to repair')

    # check a few things - this will also be a place to debug the time correction
    dt_ns = signal_dict['params']['data_utc'][0] - signal_dict['params']['shot_f_u']
    assumed_delay_from_shot_utc_to_ECH = int(61e9)

    tb = signal_dict['timebase']
    itb = np.cumsum(signal_dict['params']['diff_dimraw']) if 'diff_dimraw' in list(signal_dict['params']) else None
    debug_(pyfusion.DEBUG, 0, key="interp_timebase", msg="interpolate stair step corrupted timebase")
    if itb is None:
        print('========== Unable to repair as diff_dimraw is not stored')
        return (signal_dict['timebase'])
    else:
        return 1e-9 * regenerate_dim(itb - itb[0])[0]
    last_non_nan = np.min(np.where(np.isnan(tb))[0])
    mintb = np.argsort(np.diff(tb))
    chg = np.where(np.diff(tb) != 0)[0]
    last = np.max([c for c, ch in enumerate(chg) if ch <= last_non_nan])
    # Need to start at a change and end at a change because the start of data
    # could be anywhere in the 'cycle'
    # newtb = np.interp(range(chg[0], chg[last]+1), chg[0:last+1], tb[chg[0:37+1]])

    debug_(pyfusion.DEBUG, 3, key="interp_timebase", msg="interpolate stair step corrupted timebase") 
    return(newtb)  # signal_dict['timebase'])
