import numpy as np
def broaden(inds, data=None, dw=1):
    """ broaden a set of indices or data in width by dw each side """
    # 
    if dw!=1: print('only dw=1')
    inds_new = np.unique(np.concatenate([inds[1:]-1,inds,inds[0:-1]+1]))
    return(inds_new)

def between(var, lower, upper, closed=True):
    """ return whether var is between lower and upper 
    includes end points if closed=True
    """
    if len(np.shape(var)) == 0:
        if closed:
            return ((var >= lower) & (var <= upper))
        else:
            return ((var > lower) & (var < upper))

    else:
        avar = np.array(var)
        if closed:
            return ((avar >= lower) & (avar <= upper))
        else:
            return ((avar > lower) & (avar < upper))

bw = between
btw = between

def decimate(data, limit=None, fraction=None):
    """ reduce the number of items to a limit or by a fraction
    returns the same data every call
    """
    if (fraction == None and limit==None):
        limit=500
    if fraction != None: 
        if fraction>1: raise ValueError('fraction ({f}) must be < 1'
                                        .format(f=faction))
        step = np.max([int(1/fraction),1])
    else:
        step = np.max([int(len(data)/limit),1])
    return(np.array(data)[np.arange(0,len(data), step)])        

def his(xa):
    """ print the counts and fraction of xa binned as integers
    """
    #    xmin = np.nanmin(xa)
    #    xmax = np.nanmax(xa)
    xa = np.array(xa)
    for x in np.unique(xa):
        w = np.where(xa == x)[0]
        print('{x:3d}: {nx:10d}  {fx:10.2f}%'.
              format(x = x+0, nx = len(w), fx=float(100*len(w))/len(xa)))
