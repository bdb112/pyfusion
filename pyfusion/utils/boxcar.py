import numpy as np
def rotate(x, offs=3):
    """ 
    Works for +/- offs
    >>> rotate([1,2,3,4,5,6],2)
    [3,4,5,6,1,2]
    """
    xlist = x.tolist()
    return np.array(xlist[offs:] + xlist[0:offs])

def boxcar(sig=None, period=167, tim=None, maxnum=99999999, return_numused=False, debug=0):
    """
    period:   Actual period in samples, can be non-intergral.  If so, a
              little less than all the data is used to avoid fetching out of
              bounds.  The safety margin is more generous than it needs to be.

              signal can be a signal part of an MDS signal, or an array
    """
    # make sure that any MDS data has been converted to a true nd.array
    # test for presence of .value_of()  because nd.arrays have a (different) .data() method!xs
    if hasattr(sig, 'value_of'):  # otherwise will get 1 extra element -MDS bug ???
        sig = sig.data()

    iper = int(period)  # integer part of the period
    numcyc = len(sig) // iper
    if iper != period:
        numcyc = len(sig) // iper - 2   # take off 2 for a safety margin
        numcyc = numcyc - numcyc//iper # and allow for 1 extra point lost each cycle

    if numcyc > maxnum:
        numcyc = min(maxnum, numcyc)  # can reduce number as we do the tuning.
        if debug > 0:
            print('truncating to {numcyc} cycles'.format(numcyc=numcyc))
    elif numcyc < maxnum:
        if numcyc < 1:
            raise LookupError('less than one period in the segment')
        print('Only {numcyc} cycles in the given data segment'.format(numcyc=numcyc))

    ar2d = []
    tm2d = []
    if tim is not None:
        period_sec = period * np.diff(tim).mean()
    for cyc in range(numcyc):
        offs = int(cyc * (period - iper))  # offset grows a little each time
        ar2d.append(sig[cyc * iper + offs: (cyc + 1) * iper + offs])
        if tim is not None:
            tm2d.append(tim[cyc * iper + offs: (cyc + 1) * iper + offs] - cyc * period_sec)
        
    if tim is not None:
        tm1d = np.reshape(tm2d, np.product(np.shape(tm2d)))
        ar1d = np.reshape(ar2d, np.product(np.shape(ar2d)))
        inds = np.argsort(tm1d)
        return(tm1d[inds], ar1d[inds]) # RETURN here if tim

    # this is virtually an ELSE!
    accum = np.mean(ar2d,axis=0)
    if debug:
        from matplotlib import pyplot as plt  # should be at top, 
        plt.plot(accum)
    if return_numused:
        return(accum, numcyc)
    else:
        return(accum)
