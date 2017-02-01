"""
 nlines=3000, npts=10 rasterized, dpi=400 get  too many blocks error 
 nlines=10000, npts=10 rasterized, dpi=300 get  too many blocks error 
 nlines=10000, npts=10 rasterized, dpi=100 OK
"""
from matplotlib import pyplot as plt
import numpy as np
import os


def join_segments(*args):
    """take a list or array of sequences representing line coordinates,
    and join them using nans as the separator.  This way plotting will
    be faster, and rasterization will be more efficient.  Works on one
    or more lists, typical call is

    >>> x2, y2 = join_segments(x,y)
    >>> plt.plot(x2,y2,'r',lw=30./nlines,rasterized=True)

    """
    if len(args)>1:  # process multiple args recursively
        return([join_segments(arg) for arg in args])
    # a single arg
    veclist = []
    for row in args[0].T:
        lst = row.tolist()
        lst.append(np.nan)
        veclist.extend(lst)
    return(veclist)

#  test code
if __name__ == '__main__':

    npts = 10
    nlines = 1000
    x,y = np.mgrid[0: 1: 1j*npts, 0: 1: 1j*nlines]
    y = 0.1 * np.random.random((npts, nlines))

    #test single arg case
    x1 = join_segments(x)
    y1 = join_segments(y)

    plt.figure()
    plt.plot(x1,y1,lw=30./nlines,rasterized=True)
    plt.ylim(0,1)
    plt.show()
    plt.savefig('junk1.pdf', dpi=400)

    # test multi arg case
    x2, y2 = join_segments(x,y)
    plt.figure()
    plt.plot(x2,y2,'r',lw=30./nlines,rasterized=True)

    # test rasterized
    plt.figure()
    pt = plt.plot(x, y, 'g', linewidth=30./nlines) #,rasterized=True)
    plt.gcf().set_rasterized(True)
    plt.ylim(-1, 1)
    plt.show()
    plt.savefig('junk.pdf', dpi=200)
    os.system('evince junk.pdf')
