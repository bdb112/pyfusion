#from DA_datamining import DA
import os, pickle
import numpy as np
# Note: this would not run while the import of shot_list.py was at the top
# probably should be combined with get shot_list (as module get_shot_info)
from pyfusion.acquisition.W7X.get_shot_list import get_shotDA
from pyfusion.utils.time_utils import utc_GMT, utc_ns

# should think about avoiding reloads - but need to reload if list has changed
# maybe check the file timestamp.
def get_shot_info(date, shot, item='comment', quiet=False):
    """ return utc values for a given shot and date 
    We define shot 0 to be midnight (i.e. 00am) and shot >=999 to be 
       midnight the next day (i.e. 24:00 today) for use in valid_since

    get_shot_info(date=20160310, shot=10)

    """ 
    shotDA = get_shotDA()  # centralise access to shotDA file
    # shotDA = pickle.load(open(os.path.join(this_dir,'shotDA.pickle'),'rb'))
    ws = np.where((shotDA['date'] == date) & (shotDA['progId'] == shot))[0]
    if len(ws) == 1:
        if item == 'utc':
            return([shotDA['start_utc'][ws][0],shotDA['end_utc'][ws][0]])
        else:
            return(shotDA[item][ws])

    elif len(ws) == 0:
        if (shot == 0) or (shot >= 999):   # shot zero is assumed to be midnight
            #  We define shot 0 to be midnight (i.e. 00am) 
            #  and shot >=999 to be midnight the next day (i.e.) 24:00 today
            return(2*[utc_ns(str(date + int(shot >= 999)))]) # this is needed for valid_since
        if quiet:
            return(None)
        else:
            raise LookupError('program {s} not found on day {d}'.format(s=shot,d=date))

    else:
        raise LookupError('more than one match? to shot {shot}'
                          .format(shot=shot))
        
def get_shot_utc(shot, quiet=False):
    """ convenience routine to obtain shot utc values 
    - return unchanged if already a utc
    """
    if shot[0] > 1e9: # already utcs!
        return(shot)
    return(get_shot_info(date=shot[0], shot=shot[1], item='utc', quiet=quiet))
