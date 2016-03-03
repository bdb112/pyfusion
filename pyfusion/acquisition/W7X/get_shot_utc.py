#from DA_datamining import DA
import os, pickle
import numpy as np
    
# should think about avoiding reloads
this_dir = os.path.dirname(__file__)
# 'rb' causes a problem with winpy
shotDA = pickle.load(open(os.path.join(this_dir,'shotDA.pickle'),'r'))

def get_shot_utc(date, shot):
    ws = np.where((shotDA['date'] == date) & (shotDA['progId'] == shot))[0]
    if len(ws) == 1:
        return([shotDA['start_utc'][ws][0],shotDA['end_utc'][ws][0]])
    elif len(ws) == 0:
        raise LookupError('shot {s} not found on {d}'.format(s=shot,d=date))
    else:
        print('more than one match?')
        
    
