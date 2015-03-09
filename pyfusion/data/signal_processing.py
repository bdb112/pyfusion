""" Boyd's python for stand alone, general signal processing, try to be 
"efficient".  
Stuff that imports pyfusion should live in pyfusion_sigproc.py.  Put stuff here so that recompilation doesn't require restarting pyfusion.
"""
from numpy import cumsum, double, arange, array, shape, sqrt, cos, sin, size, shape
import numpy as np
import pylab as pl

VERBOSE=2

def splot(signal, *args, **kwargs):
    """ simple wrapper to plot to accept tuples of form (timebase, data)
    """
    if len(signal) == 2:
        if len(shape(signal)) == 1: pl.plot(signal, *args,  **kwargs)
        else: pl.plot(signal[0], signal[1], *args,  **kwargs)
    else:
        pl.plot(signal, *args, **kwargs)

def smooth(data, n_smooth=3, timebase=None, causal=False, indices=False, keep=False):
    """ An efficient top hat smoother based on the IDL routine of that name.
    The use of cumsum-shift(cumsum) means that execution time
    is 2xN flops compared to 2 x n_smooth x N for a convolution.  
    If supplied with a timebase, the shortened timebase is returned as
    the first of a tuple.  
    
    causal -- If true, the smoothed signal never preceded the input,
              otherwise, the smoothed signal is "centred" on the input 
              (for n_smooth odd) and close (1/2 timestep off) for even
    indices -- if true, return the timebase indices instead of the times       
    data = (timebase, data) is a shorthand way to pass timebase
    n_smooth - apply recursively if an array  e.g. n_smooth=[33,20,14,11] 
               removes 3rd, 5th, 7th, 9th harmonics
               fraction values are interpreted as timeintervals.

    >>> smooth([1,2,3,4],3) 
    array([ 2.,  3.])
    >>> smooth([1.,2.,3.,4.,5.],3) 
    array([ 2.,  3.,  4.])
    >>> smooth([1,2,3,4,5],timebase=array([1,2,3,4,5]),n_smooth=3, causal=False)
    (array([2, 3, 4]), array([ 2.,  3.,  4.]))
    >>> smooth([0,0,0,3,0,0,0],timebase=[1,2,3,4,5,6,7],n_smooth=3, causal=True)
    ([3, 4, 5, 6, 7], array([ 0.,  1.,  1.,  1.,  0.]))
    >>> smooth([0,0,0,3,0,0,0],timebase=[1,2,3,4,5,6,7],n_smooth=3, causal=True, indices=True)
    ([2, 3, 4, 5, 6], array([ 0.,  1.,  1.,  1.,  0.]))
    >>> smooth([0,   0,   0,   0,   5,   0,    0,  0,   0,   0,   0], 5, keep=1)
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0., -1., -1.])

    Last example: keep=1:
        Better to throw the partially cooked ends away, but if you want to
    keep them use keep=True.  THis is useful for quick filtering
    applications so that original and filtered signals are easily
    compared without worrying about timebase
    """
    if len(shape(n_smooth))==1:
        n_sm = n_smooth[0]
    else:
        n_sm = n_smooth
    if len(data) == 2:  # a tuple (timebase, data)
        timebase=data[0]
        data = data[1]
    if n_sm < 1:  # assume it is an interval
        if timebase is None:
            raise ValueError('fractional n_smooth {n_sm} implies a smoothing '
                             ' time interval, but no timebase given'
                             .format(n_sm-n_sm))
        else:
            n_sm = int(0.05 + n_sm/np.average(np.diff(timebase)))

    csum = cumsum(data)
    if not(keep):
        sm_sig = csum[n_sm-1:]
        # just the valid bit
        #sm_sig[1:] -= csum[0:-n_sm]  #  should work?  - bug in python?
        sm_sig[1:] = sm_sig[1:] - csum[0:-n_sm]
        sm_sig=sm_sig/double(n_sm)
    else:   # try to retain full length for quick tests - only allow causal=False
        if causal: raise ValueError, "can't be causal and keep ends"
        else:
            sm_sig=array(data)*0  # get a vector of the right length
            half_offs = n_sm/2   # use integer round down, only perfect for odd n
            # See test code [0, 0, 0, 1, 0, 0, 0]
            # for the n=3 case, want sm[3] to be cs[4]-cs[1] so load cs[1:] into sm, 
            # then subtract cs[0] from sm[2] etc
            # for n=5, want sm[5] to be cs[7]-cs[2] so load cs[2:] into sm 
            # and subtr cs[0] from sm[3]
            sm_sig[0:-half_offs] = csum[half_offs:]  # get the first bit left shifted
            sm_sig[half_offs+1:] = sm_sig[half_offs+1:] - csum[0:-(half_offs+1)] 
            sm_sig=sm_sig/double(n_sm)
    if timebase is None: 
        if size(n_smooth) > 1:
            return(smooth(sm_sig, n_smooth[1:],timebase=timebase, causal=causal, indices=indices, keep=keep))
        else:
            return(sm_sig)
    else:
        if causal: 
            offset = n_smooth-1
        elif keep: 
            offset = 0
        else:
            offset = n_smooth/2

        if indices:
            sm_timebase = range(offset,offset+len(sm_sig))
        else:
            sm_timebase = timebase[offset:offset+len(sm_sig)]

        if size(n_smooth) > 1:
            (ttb, tsig) = smooth(sm_sig, n_smooth[1:],timebase=sm_timebase, 
                                 causal=causal, indices=indices, keep=keep)
            return((ttb, tsig))
        else:
            return((sm_timebase, sm_sig))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import time
    from numpy import convolve
    st=time.time()
    ntest=1e5
    loops=100
    xn=smooth(arange(ntest),7)
    xt=smooth(arange(ntest), timebase=arange(ntest)/100,n_smooth=.07)
    print('fractional test '),
    if (xn==xt[1]).all(): print('OK')
    else:
        print('failed: results don''t match', xn,xt[1])

    for n in arange(loops): x=smooth(arange(0,ntest,1.),20)
    dt=time.time()-st
    print("%.3g ns/number" % (1e9*dt/ntest/loops)),

    st=time.time()
    x=convolve(arange(0,1e6,1.), [1,1,1])
    print("for smooth(20) compared with convolve: \n%.3g ns" % ((time.time()-st)*1e9/len(x))),
    x=convolve(arange(0,1e6,1.), np.ones(20))
    print("for smooth(3) and %.3g ns for smooth(20)" %
          ((time.time()-st)*1e9/len(x)))
## 35ns compared to 25 ns for IDL - not bad! (but overhead adds 10% at array size of 1e3!)
## st=systime(1)& for i=1,10 do x=smooth(findgen(1e6),3) & en=systime(1) & print, 1e9/10*(en-st)/n_elements(x)

def smooth_n(data, n_smooth=3, timebase=None, causal=False, iter=3, keep=False,
             indices=False):
    """ Apply smooth "iter" times.
    [ smooth() doc follows: ]
    """
    tim=timebase
    dat=data
    if timebase != None: 
        for i in range(0,iter):
            tim, dat = smooth(dat, n_smooth=n_smooth, timebase=tim, 
                             causal=causal, keep=keep, indices=indices) 
        return((tim,dat))   
    else:
        for i in range(0,iter):
            dat = smooth(dat, n_smooth=n_smooth, timebase=timebase, 
                             causal=causal, keep=keep, indices=indices) 
        return(dat)   
        
smooth_n.__doc__ += smooth.__doc__

def cross_correl(x1, x2, nsmooth=21, n_times=3):
    """ <x1.x2>/sqrt(<x1.x1> * <x2,x2>) averaged over nsmooth points n_times
    Ideally extract raw data from pyfusion signals, but that makes it pyfusion
    specfic - bad.
    """
    xc = smooth_n(x1*x2,nsmooth, iter=n_times)/sqrt(smooth_n(x1*x1,nsmooth, iter=n_times)*smooth_n(x2*x2,nsmooth,iter=n_times))
    
    if VERBOSE>6: 
        import pylab as pl
        pl.plot(xc,'.',hold=0,markersize=0.1)

    return(xc)    

def powerof2(n, near=False):
    from numpy import log2
    p = log2(n+0.1)
    return(2**int(p))

def analytic_phase(x, t=None, subint=None):
    """ gets the phase from an amazing variety of signals
    http://en.wikipedia.org/wiki/Analytic_signal
    subinterval idea is not debugged and is probably unnecessary
    may shorten data?
    """
    from scipy.fftpack import hilbert
    from pyfusion.utils import fix2pi_skips
    from numpy import zeros, arctan2
    # this subinterval idea does not seem necessary and is not debugged
    if subint != None:
        subint=powerof2(subint)
        nsubints = int(len(x)/subint)
        phs = zeros([nsubints, subint])
        for i in range(nsubints):
            xsub = x[i*subint:(i+1)*subint]
            phs[i,:] = arctan2(xsub, hilbert(xsub))
        phi = phs.flatten()
    else:
        y=hilbert(x)  # use hilbert twice just to remove DC (lazy)
        phi = arctan2(y, hilbert(y))

    return(fix2pi_skips(phi, sign='+'))

def test_analytic_phase(verbose=3):
    """ 
    >>> test_analytic_phase()
    """
    t=array(range(10000))/100.
    x=sin(t+t**2/30)
    y=cos(t+t**2/30)
    if len(analytic_phase(x)) < len(x):  
        print('shortened from {0} to {1}'
              .format(len(x),len(analytic_phase(x)))) 
    if verbose>2:
        xx0=pl.plot(smooth(analytic_phase(x) - analytic_phase(y), 10), hold=0)
        xx1=pl.plot(smooth(analytic_phase(x) - analytic_phase(y+100), 10),
                     'k--', hold=1)
        pl.title('black dash and blue should overlay well')
        if not(pl.isinteractive()): pl.show()
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
