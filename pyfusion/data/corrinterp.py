""" cross correlation for signals on different timebases
   This version has no MDS or pyfusion dependence - see also pyfusion_sigproc.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def correlation(x, y, tx=None, ty=None, AC=True, coefft=True):
    """  AC => remove means first
         coefft: True is dimensionless - else return the amplitude
         Uses the coarsest of the two timebases
    >>> t1 = np.linspace(0,50,100)
    >>> t2 = np.linspace(0,50,15)
    >>> s1 = np.sin(t1)
    >>> s2 = np.sin(t2)
    >>> round(correlation(s2, 0.1*s1, tx=t2, ty=t1, coefft=0)[0], 4) # avoid 0.09800001, don't use np
    0.0978
    >>> round(correlation(s1, 0.1*s2, tx=t1, ty=t2, coefft=0)[0], 4) # coarse grids chosen to show detail
    9.7815
    >>> round(correlation(s1, 0.1*s1, tx=t1, ty=t1, coefft=0)[0], 4) # make sure it can cope if interp not needed
    0.1
    """

    clate = np.correlate
    mean = np.mean
    if tx is not None:
        if (ty is None) ^ (ty is None):
            raise ValueError('both tx and ty should be defined or None')
        # ensure tx is the coarser grid, and the one used for result
        if np.diff(tx).mean() < np.diff(ty).mean():
            # print('swap')
            x, y = y, x
            tx, ty = ty, tx

        win = np.where((tx >= np.min(ty)) & (tx <= np.max(ty)))[0]
        if len(win) != len(x):
            print('Warning - reducing to common time range')
            x = x[win]
            tx = tx[win]

        fint = interp1d(ty, y)
        y = fint(tx)

    if AC:
        x = x - mean(x)
        y = y - mean(y)
    if coefft:
        return(clate(x, y)/np.sqrt(clate(x, x) * clate(y, y)))
    else:
        return(clate(x, y)/np.sqrt(clate(x, x) * clate(x, x)))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
