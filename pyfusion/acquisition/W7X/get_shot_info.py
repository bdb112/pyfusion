#from DA_datamining import DA
import os, pickle
import numpy as np
from .get_shot_list import get_shotDA
    
# should think about avoiding reloads - but need to reload if list has changed
# maybe check the file timestamp.
def get_shot_info(date, shot, item='comment', quiet=False):
    """ return utc values for a given shot and date 
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
        if quiet:
            return(None)
        else:
            raise LookupError('shot {s} not found on {d}'.format(s=shot,d=date))

    else:
        raise LookupError('more than one match? to shot {shot}'
                          .format(shot=shot))
        
def get_shot_utc(date, shot, quiet=False):
    """ convenience routine to obtain shot utc values """
    return(get_shot_info(date=date, shot=shot, item='utc', quiet=quiet))

