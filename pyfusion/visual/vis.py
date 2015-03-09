import pylab as pl
import numpy as np

def vis(dd, ax=None, size=1, show=False, inds = None, debug=False):
    """ return the indices in dd which are currently visible and greater
    in size than size (>=).  Use after visual.sp  
    size = 0 will return all points and sizes
    """

    from pyfusion.data.convenience import btw

    if inds is None:  raise Exception('inds must be set  (-1 for all) for now')

    if ax is None: 
        ax=pl.gca()

    x = ax.get_xlabel()
    xlim = ax.get_xlim()
    y = ax.get_ylabel()
    ylim = ax.get_ylim()

    if len(np.shape(inds)) == 0 and inds == -1:
        inds = np.arange(len(dd[x]))

    #  w = np.where(btw(dd[x],xlim) & btw(dd[y],ylim))[0]
    w = np.where(btw(dd[x][inds],xlim) & btw(dd[y][inds],ylim))[0]
    if len(w) == 0:
        raise LookupError('No points found')
    if show:
        pl.scatter(dd[x][w], dd[y][w],'+')

    if debug: 1/0
    return(w)

    
