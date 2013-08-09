import numpy as np
import pylab as pl
import pyfusion
from pyfusion.debug_ import debug_

""" 
Tests:
Memory: 10M 2 element tuple: -> 1.3GB 130b/elt, 10M 14 elt 2.2G = 220 b/elt
  for i in range(10000000): d.update({(i,10,):i})
  12x8 = 96 bytes extra = about right.
  
  Precalculate random indices:
  10M x14 in range 0-100 takes 110 secs and 1.9 gig to calc, and 12 sec to 
  put in dict, occupying 1gB (as monitored by ubuntu)
inds = []; d={}
time for i in range(2*1000*1000): inds.append(tuple(array(random.random(14)*100,dtype=int).tolist()))
time for ind in inds: d.update({ind:ind[0]})
2M, tuple -> 1340M.  cf 117M 
  add another 10M by reversing tuples.  +1.3GB - about 130 bytes/entry
  this compares with 1100M/2 million for on the fly calculation (550 bytes/ent).
time for i in range(2*1000*1000): dd.update({tuple(random.random(14).tolist()):i}
n_bins = 60
eps=.001

pmax=256  # scale factor separating bins


# this approach vectorises over instances AND signals but uses too
# much space in the conversion to indices 
scalevec=np.tile(np.logspace(0,16,9),2)
scalemat=np.tile(scalevec[0:nsig],[ninsts,1])

inds=np.zeros((ninsts,2),dtype=int)
fact = n_bins/(2*(np.pi+eps))
inds[:,0] = np.array(sum((scalemat*(phases+np.pi)*fact)[:,0:9],1),dtype=int)
inds[:,1] = np.array(sum((scalemat*(phases+np.pi)*fact)[:,9:],1),dtype=int)

inds = [tuple(p) for p in inds]
for ind in inds: d.update({ind:ind[0]})

"""

# there is really no need for a class, unless we want to check bounds
# but that is a good reason!
class CoordHD():
    def __init__(self, dims, debug=0):
        self.dims = dims
        self.d = {}
        self.debug = debug

    def get(self, *indices):
        """ get(1,2,3)  args not a tuple """
        if self.debug>0: print('get indices', indices)
        for (i, index) in enumerate(indices):
            if index>= self.dims[i]:
                raise LookupError()
        if not self.d.has_key(indices): return(0)
        return(self.d[indices])
    
    def set(self, indices, val):
        """ set((1,2,3),49.)  """
        for (i, index) in enumerate(indices):
            if index>= self.dims[i]:
                raise LookupError()
        self.d[indices] = val


# This version uses strings as indicies, and doesn't check bounds on the fly
# Instead there is a check_bounds rouitine which can be run after the fact  
# each index may span up to 255.      
class CoordHDs():
    def __init__(self, dims, debug=0):
        self.dims = dims
        self.d = {}
        self.debug = debug

    def get(self, indices):
        """ string version doesn't require the tuple notation (*)  """
        if self.debug>0: print('get indices', indices)
        if not self.d.has_key(indices): return(0)
        return(self.d[indices])
    
    def set(self, indices, val):
        """ set(inds,49.)  """
        self.d[indices] = val


def find_eps(x,value=None):
    """ find the smallest number that will always exceed the representational
    accuracy of x in the range around value.
    """
    num = 10000

    xx = np.repeat(x.copy(), num)
    if value is not None: 
        xx[:] = value
    xxt = np.repeat(x.copy(), num)
    t_eps = np.logspace(-50,-1, num)

    xxt[:] = xx[:] + t_eps[:]
    wgt = np.where(xxt>xx)[0]

    xxt[:] = xx[:] - t_eps[:]
    wlt = np.where(xxt<xx)[0]

    if (len(wgt)==0 or len(wlt)==0): 
        raise Exception("can't determine precision of {x}"
                        .format(x=x))
    
    return(t_eps[wlt[0]]+t_eps[wgt[0]])


def histogramHD(d, bins=None, method='safe'):
    """ make a histogram of data too high in dimension to use histogramdd,
    by successive calls to histogramdd().
    The simplest implementation is to assume all bins are equal and in the 
    range -pi..pi subdivided into nbins bins.
    
    First version will use the coo_utils for simplicity  - not sure how 
    efficient lookup is though - and there is no obvious way to make an 
    array larger than memory
    
    Second - use a dictionary to store the results - 120ns lookup in a 100k dict
    Assume we can do a rank R array with histogramdd
    Should really store the bin boundaries for later reference
    """

    """
    higher speed potential (but need to do in chunks) - large interm. result
    timeit xxx=bytearray(array((ph5+3.14)*4.33,dtype=int8)) 1.05us/instance
    timeit str(xxx[1000])  225ns
    """ 

    (n_instances, n_signals) = np.shape(d)
    maxdim_dd = 3
    if bins is None: bins = 12
    eps = find_eps(d[0,0],value=np.pi)
    # note - at present, this chunk (bins) is not used (speed)
    if np.isscalar(bins): 
        n_bins = bins
        # small overlap to allow for finite precision (only matters for float16)
        bins = np.pi * np.linspace(-(1+eps), 1+eps, bins)  

    # For each cell, we need the number that lie in that range for each signal
    # for each dimension
    #    for each bin 
    """   
    In 2D it is easy:
    for ix in xbins
       for iy in ybins
          how many in this bin - record
    By induction  - to add an extra phase, for each HDD, how many are in each new bin
    SO you add a layer for each bin, and the totals in the original are subdivided
    So the function should add hyperlayers.  Problem - to extend

    # this code allows for arbitray bin edges - not really necessary here.
    # for each signal, we need to record a count of the number that are in each bin
    mask = np.empty((n_instances, n_bins), dtype=bool)
    for s in range(n_signals):
        for (i, bin) in enumerate(bins[1:]):
            mask[:,i] = ((phases[:,s]>=bins[i-1]) & (phases[:,s]<bins[i]))
        w = np.where(mask.any(1))[0]
        print(s,len(w))
    """
    """

    The only way feasible with 20^10 cells is to place them in cells one by one
    can't afford to operate bin by bin as there are bins**n_phases - say 20**10 = 10^13

    """    
    
    dims = n_signals*(n_bins,)
    fact = n_bins/(2*(np.pi+eps))


    if method=='safe':
        hdd = CoordHD(dims)
        hdd.eps=eps
        print('histogramming (int index) {ni:,d} instances into {s:.2g} bins'.
              format(ni = n_instances, 
                     s=np.product(np.array(dims).astype(float))))
        for phase in d[:]:
            inds = tuple(np.array((phase+np.pi)*fact,
                                  dtype=int))
            hdd.set((inds), hdd.get(*inds) + 1)

    else:
        #string method - 2GB for 16M x 16 in 270 secs (E4300)  best so 
        # far (no-check store - no increment)
        """
        fact = n_bins/(2*(np.pi+eps))
        inds = [str(bytearray(np.array((phase+np.pi)*fact,dtype=np.int8))) 
                for phase in phases]
        d={}
        for ind in inds: d.update({ind:ind[0]})

        # do it on the fly takes 1.5G for 16Mx16
        time for phase in phases:  d.update({str(
                    bytearray(np.array((phase+np.pi)*fact,dtype=np.int8))):1})
        """
        hdd = CoordHDs(dims)
        print('histogramming (str index) {ni:,d} instances into {s:.2g} bins'.
              format(ni = n_instances, 
                     s=np.product(np.array(dims).astype(float))))

        for phase in d[:]:
            # see higher speed version in comment below the doc
            #~14us suprisingly takes 3us for str(bytarr, 3-4us each for + and *
            inds = str( bytearray(np.array((phase+np.pi)*fact,dtype=np.int8)))
            # I think this is ~ 5us
            hdd.set((inds), hdd.get(inds) + 1)
        
        """
    # this is a complicated but faster executing version, which
    # seems to consume less memory, but expanding the dictionary
    # only in bursts, with no calculations in between updates.
    # misses counts which occur more than once in the same batch.
    batch_size = 10000  # 2.3 sec for 100kx16 inds, 2.7sec for update
    for batch in range(1 + (len(d)/batch_size)):
        print(batch)
        last = min((batch+1)*batch_size, len(d))
        inds_list = [tuple(np.array((phase+np.pi)*n_bins
                                    /(2*(np.pi+eps)),dtype=int))
                     for phase in d[batch*batch_size:last]]
        #for inds in inds_list: hdd.set((inds), hdd.get(*inds) + 1)
        #for inds in inds_list:hdd.d.update({inds:1}) 1.25sec fo 1Mx16!!
        # starts at 5secs/100k, 23 secs for 100K at 
        # end still 4.6GB, 15minutes for 4MB (note - as implemented misses 
        # counts in the same batch - very important.
        vals = []
        for inds in inds_list:
            #debug_(max(pyfusion.DEBUG, hdd.debug), key='histogramHD')
            if hdd.d.has_key(inds): 
                vals.append(hdd.d[inds])
            else: 
                vals.append(0)
        for (i,inds) in enumerate(inds_list):
            hdd.d.update({inds: vals[i]+1})
    """        

    return(hdd)

# make inds list - 23usec   
# update line 27us (16 elt)
# simple update 1.2us
# simple dict access 0.5 sec
# hdd.get 14us
# so most of the time is in subscript maths and list creation
# 4Mx16 4.1GB 355 sec - same either way.

if __name__ == '__main__':

    print('CoordHD test')
    arrHD = CoordHD((4,5,6))
    arrHD.set((1,2,3),3)
    print(arrHD.get(1,2,3))

    """  n_signals n_bins n_instances  time
              6      12     100k        3.5 sec
              16     12     100k        5.2 sec  
    """

    print('generate artificial data')
    from numpy.random import random as randu
    n_signals = 6
    n_instances = 100000
    n_bins = 12
    seed = 0

    np.random.seed(seed)
    phases = 2*np.pi * (np.random.random((n_instances, n_signals)) - 0.5)
    # the normal way
    try:
        hdd = np.histogramdd(phases, n_signals*[np.pi * np.linspace(-1,1,n_bins+1)])
        flat_counts = hdd[0].flatten()
        print('histogramdd gives {b:.2g} bins, {s} set, total of {t}'
              .format(b=len(flat_counts), s=len(np.where(flat_counts!=0)[0]),t=np.sum(flat_counts)))
    except (ValueError,MemoryError):
        print('Too big for histogramdd')
        
    # the high dimensional way
    sel = range(0,10)
    sel.extend(range(12,16))
    hHD = histogramHD(phases,bins = n_bins)
    phs = hHD.d.keys()
    counts = [hHD.d[k] for k in hHD.d.keys()]
    print('{n} elements set, total of {t}, max count of {m}'
          .format(n=len(counts), t=np.sum(counts), m=np.max(counts)))

    """ 
    n_bins = 48
    sel = range(0,10)
    sel.extend(range(12,16))
    hHD = histogramHD(phases[:,sel],bins = n_bins)
    phs = hHD.d.keys()
    counts = [hHD.d[k] for k in hHD.d.keys()]
    inds = argsort(counts)
    for i in inds[-100:]: plot(-np.pi+2*np.pi/n_bins*np.array(phs[i]), linewidth=0.001*n_bins*counts[i])
    show()

    """
