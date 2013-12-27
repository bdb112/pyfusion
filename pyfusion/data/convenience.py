import numpy as np
import os

def broaden(inds, data=None, dw=1):
    """ broaden a set of indices or data in width by dw each side """
    # 
    if dw!=1: print('only dw=1')
    inds_new = np.unique(np.concatenate([inds[1:]-1,inds,inds[0:-1]+1]))
    return(inds_new)

def between(var, lower, upper=None, closed=True):
    """ return whether var is between lower and upper 
    includes end points if closed=True
    alternative call is between(var, range)   e.g. between(x, [1, 2])
    """
    # want to catch arg3 given when arg2 is a range
    if (len(np.shape(lower))==1 and np.shape(lower) == (2) and 
        np.shape(lower) != np.shape(var)):
        if upper is None:
            upper = lower[1]
            lower = lower[0]
        else:
            raise ValueError(' if arg 2 is a range, arg3 must not be given')

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

def his(xa, tabs=False, sort=-1):
    """ print the counts and fraction of xa binned as integers
    sort=1,-1 sorts by most frequent (first, last), 
    """
    #    xmin = np.nanmin(xa)
    #    xmax = np.nanmax(xa)
    xa = np.array(xa)
    xarr,nxarr = [],[]
    for x in np.unique(xa):
        w = np.where(xa == x)[0]
        xarr.append(x+0)
        nxarr.append(len(w))

    if sort!=0: 
        ii = np.argsort(nx)
        if sort > 0 : ii=ii[::-1]  # reverse so that greatest is first
    else: ii = range(len(nx))
    for i in ii:
        # fails to generate tabs (\t)- is it the terminal software that detabifies?
        # Use a single space instead - soffice doesn't combine spaces.
        if tabs:
            fmt = '{x:0d}: {nx:0d} {fx:.2f}%\n'
        else:
            fmt = '{x:0d}: {nx:10d}  {fx:10.2f}%\n'
        os.write(1,fmt.          # I had hoped os.write would be "raw" - but not
                 format(x = xarr[i]+0, nx = nxarr[i], 
                        fx=float(100*nxarr[i])/len(xa)))
