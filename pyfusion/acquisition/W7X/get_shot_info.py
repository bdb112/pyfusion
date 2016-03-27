#from DA_datamining import DA
import os, pickle
import numpy as np
    
# should think about avoiding reloads - but need to reload if list has changed
# maybe check the file timestamp.
this_dir = os.path.dirname(__file__)
# 'rb' causes a problem with winpy - maybe using protocol 2 will fix?
#  under proto 2, need ,encoding='ascii' in python3 if file written by python2

def get_shot_info(date, shot, item='comment', quiet=False):
    shotDA = pickle.load(open(os.path.join(this_dir,'shotDA.pickle'),'rb'))
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
    return(get_shot_info(date=date, shot=shot, item='utc', quiet=quiet))
