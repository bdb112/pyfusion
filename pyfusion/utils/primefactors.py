""" find prime factors of a number, especially to allow estimation of the
time taken by an FFT

For fast methods up to ~ 10^6 (e.g. numpy
http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python

For methods with good scaling:
http://stackoverflow.com/questions/2211990/how-to-implement-an-efficient-infinite-generator-of-prime-numbers-in-python/

primesfrom2to3 (numpy is fastest for 1e9: 9secs, and rwh_primes2 47s pure
see http://www.fftw.org/speed
MFlops ~ 5*N*log2(N)/fft_time_estimate(N)/1e6 for complex fft ~ 1399 for 32768
fftw3 is ~ 2x faster than fftw2 (more for large arrays, less for small (<100k elts)

These initial attempts were not efficient - pyfftw is much better
speedup is 3-10x
/bin/fftw3bench cf$((32768+37)) 1.08ms on E4300 fftw3, 1.66 for numpy
improve a little with "exhaustive" on lenny, but no improvement on laptop
 ./bench -oexhaustive cif$((32768+37))
Problem: cif32805, setup: 34.59 s, time: 974.75 us, ``mflops'': 2524.4

larger FFT
lenny: cif$((32768*256)) 519.58 ms, ``mflops'': 1856.7 (8M samples)
E4300                    724.98 ms, ``mflops'': 1330.6

IDL 8 fft is similar speed, but not for some odd sizes (e.g. 2**15+37)

Speeds with ./configure --enable-sse2 
bdb112@lenny:/tmp/fftw3/fftw-3.3.3/tests$ ./bench cif$((32768))
Problem: cif32768, setup: 1.79 s, time: 569.31 us, ``mflops'': 4316.8
bdb112@lenny:/tmp/fftw3/fftw-3.3.3/tests$ ./bench -onosimd cif$((32768))
Problem: cif32768, setup: 1.12 s, time: 816.06 us, ``mflops'': 3011.5
bdb112@lenny:/tmp/fftw3/fftw-3.3.3/tests$ ./bench -onosimd cif$((32768*32))
Problem: cif1048576, setup: 2.34 s, time: 50.32 ms, ``mflops'': 2083.9
bdb112@lenny:/tmp/fftw3/fftw-3.3.3/tests$ ./bench cif$((32768*32))
Problem: cif1048576, setup: 2.00 s, time: 39.95 ms, ``mflops'': 2625

./configure --enable-sse2  --with-combined-threads --enable-threads
2 threads -> 1.65x - 3 a tiny bit faster, 4 same as 2threads
Problem: cif1048576, setup: 1.93 s, time: 40.18 ms, ``mflops'': 2609.6
bdb112@lenny:/tmp/fftw3/fftw-3.3.3$ tests/bench -onthreads=2 cif$((32768*32))
Problem: cif1048576, setup: 1.68 s, time: 24.78 ms, ``mflops'': 4230.7

8M elts 2 is 1.8x faster, 4 no better.
bdb112@lenny:/tmp/fftw3/fftw-3.3.3$ tests/bench -onthreads=1 cif$((32768*256))
Problem: cif8388608, setup: 96.84 s, time: 361.73 ms, ``mflops'': 2666.9
estimate only gives 486 ms
bdb112@lenny:/tmp/fftw3/fftw-3.3.3$ tests/bench -onthreads=2 cif$((32768*256))
Problem: cif8388608, setup: 72.35 s, time: 204.83 ms, ``mflops'': 4709.6

-nosimd c128 (r64,i64) 50ms for cif1048576 cf 37ms.  c64 is 1.9x faste -> 27ms

""" 

import numpy as np
import math

def primesfrom2to2(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    import numpy
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(n/3+((n+1)&1)*(n%3)/2, dtype=numpy.bool)
    sieve[0] = False
    for i in xrange(int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1+(i&1)
            sieve[      ((k*k)/3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))/3::2*k] = False
    return numpy.r_[2,3,((3*numpy.nonzero(sieve)[0]+1)|1)]


def simple_primefactors(x):
    """ used for testing faster versions 
    6us to factorize 2**20, 121ms to factorise 2**20-3 (prime)
    """
    factorlist=[]
    loop=2
    while loop<=x:
        if x%loop==0:
            x/=loop
            factorlist.append(loop)
        else:
            loop+=1
    return factorlist

def primefactors(x, limit=2**63):
    """ This is meant to be efficient for fft_time_estimation.
    14us for 2**20 298us for 2**20-3
    """
    # with np.sqrt and % 100, 363us for 2**20-3
    # with math.sqrt and %15  298us for 2**20-3    14.9 for 2*20
    factorlist=[]
    loop=2
    while loop<=min(limit, x):
        if x%loop==0:
            x/=loop
            factorlist.append(loop)
        else:
            if loop==2:
                loop+=1
            else: loop+=2
        # try to avoid taking sqrt too often
        if loop>limit or (((loop-1)%15)==0 and loop>math.sqrt(x)):
            factorlist.append(x)
            return(factorlist)

    return factorlist

def fft_time_estimate(n, fft_type='numpy'):
    """ time in sec """
    # FFT is supposed to be N.log N, (FMM) is O(n)
    facts = primefactors(n)
    if fft_type == 'numpy':
        time_cal = 2.6e-10 #  ohead-7, oheadlog=1 E4300, float64 complex fft numpy
        ohead,oheadlog = 34,-.5 
    elif fft_type=='fftw3':
        time_cal = 1.9e-13
        ohead,oheadlog = 2.3e4,-1
        print('Warning- this is a lousy fit!')

    else: raise ValueError('fft_type {f} not understood'.format(f=fft_type))

    # ohead=7 and +1 are empirical - mid range (100*t0) time estimates are high 
    # replace with optimised values - a lot better
    cost = n*np.sum((ohead+np.array(facts)) * np.log((np.array(facts)+oheadlog)))
    return(time_cal * cost)

def nice_FFT_size_above(n,max_iterations=None):
    """ 
    Note: Using numpy.fft, it may be better to use the more accurate 
    "fft_time_estimate" above
    assume cost is \Product{N x log(N)} with Ni the factors of n.
    default max_iterations is ln(N)

    See also next_nice_number in filters.py - more general and simpler
    """
    if max_iterations == None:
        max_iterations = 10*int(np.log(n))
    lowest_cost = 9e99 ; best_n = 0
    for nn in range(n, n+max_iterations):
        cost = fft_time_estimate(n)
        print('cost estimate for {n} = {c:.2f}'.format(n=nn, c=cost))
        if cost < lowest_cost:
            lowest_cost = cost
            best_n = nn

    print('cost estimate for {n} = {c:.2f}'.format(n=best_n, c=lowest_cost))
    return(best_n)
        
def optimise_fit(atimes):
    from scipy.optimize import leastsq
    # try to optimise the fit paramters.
    # following http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

    def residuals(p, y, x):
        (time_cal, ohead, oheadlog) = p
        # obtain a list of estimates for each array size (atimes[0])
        ests = []
        for N in atimes[0]:
            facts = primefactors(int(N))
            ests.append(time_cal*N*np.sum(
                    (ohead + facts)*np.log(oheadlog + facts)))
            
        error = np.array(atimes[1]-ests)/atimes[1]
        print(p, error)
        return(error)

    p0 = [4e-4, 7, 1]
    plsq = leastsq(residuals, p0, args=(atimes[0], atimes[1]))
    print(plsq)

def get_fftw3_speed(arr, iters=10, direction=None, dtype=np.float32, **kwargs):
    """ measure the fftw3 speed for various data sizes by using
    plan with estimate, and running one instance.  If arr is int,
    then the elements are different array sizes, otherwise use the
    array.
    direction default - just fwd - use 'both' for both
    To "train": - the print allows you to ^c out.
    from pyfusion.utils.primefactors import get_fftw3_speed
    from pyfusion.utils.fftw3_bdb_utils import save_wisdom, load_wisdom
    from pyfusion.data.filters import next_nice_number
    from pyfusion.utils.primefactors import get_fftw3_speed
    for n in next_nice_number(None):
        print(n); 
        get_fftw3_speed(n, flags=['FFTW_MEASURE'],direction='both',dtype=np.float32)
    save_wisdom()

    Accepts all pyfftw.FFTW args e.g. planning_timelimit
    """
    import pyfftw
    from time import time as seconds
    if np.isscalar(arr): arr = np.array([arr])
    if np.issubdtype(arr.dtype, int):
        atimes = []
        for n in arr:
            atimes.append([n, get_fftw3_speed(np.ones(n, dtype=dtype),
                                              direction=direction, iters=iters, **kwargs)])
        return(np.array(atimes))
    else:  # do one example
        build_kwargs = dict(flags=['FFTW_ESTIMATE'])
        build_kwargs.update(kwargs)
        simd_align =  pyfftw.simd_alignment  # 16 at the moment.
        arr = pyfftw.n_byte_align(arr,  simd_align)
        out =  pyfftw.n_byte_align(np.ones(len(arr)/2+1, dtype=np.complex64), 
                                   simd_align)
        fwd = pyfftw.FFTW(arr, out, **build_kwargs)
        if direction == 'both':
            rev = pyfftw.FFTW(out, arr, direction='FFTW_BACKWARD', **build_kwargs)
        st = seconds()
        for i in range(iters):
            fwd.execute()
            if direction == 'both':
                rev.execute()
        return((seconds()-st)/iters)

if __name__ == "__main__":
    import timeit 
    import pylab as pl

    # test the primefactor code

    atimes = np.array([[  3.27680000e+04,   3.27690000e+04,   3.27700000e+04,
                          3.27710000e+04],
                       [  1.90901756e+03,   2.06589699e+04,   9.19699669e+03,
                          3.41683602e+06]])

    #optimise_fit(atimes)

    maxprime = 2**20-3
    primes = primesfrom2to2(maxprime+1)
    for (i,n) in enumerate(np.concatenate((
                np.random.randint(1,int(1e5),100),
                [2,3,37,2**20,maxprime],
                np.random.randint(1,int(maxprime),1000)))):

        # simple_primefactors has is slow for >1e5
        if i<110 and (simple_primefactors(n) != primefactors(n)):
            raise ValueError('incorrect factorization of {n}: {fbad}, cf {f}'
                             .format(n=n, f=simple_primefactors(n),
                                     fbad = primefactors(n)))
        if np.product(primefactors(n)) != n:
            raise ValueError('incorrect factorization of {n}: product = {p}'
                             .format(n=n,
                                     p = np.product(primefactors(n))))
        for f in primefactors(n):
            if f not in primes:
                raise ValueError(
                    'incorrect factorization of {n}: {f} is not prime'
                             .format(n=n,f = f))                
                             
    # test the fft_timing code
    times = []
    base_size=8192*4
    nvals = base_size-4+np.arange(19) # this includes two prime numbers (4s ea.)
    #nvals = base_size+np.arange(4) # this includes one prime number (4s ea.)
    estimate_total =  np.sum(np.array(
            [fft_time_estimate(int(i)) for i in nvals]))
    # want 1 sec total
    iters = 3   # 100 good for ~256, 3 for 8192
    iters = max(int(1./estimate_total),1)
    if estimate_total*iters>2: 
        print("prime factor tests successful, probably will take ~{t} more secs"
              .format(t=int(estimate_total*iters)))

    stmt = "y=np.fft.rfft(x)"
    for i in nvals:
        timer = timeit.Timer(stmt,
                             "import numpy as np\nx=np.arange({n})".format(n=i))
        times.append([i,1e6*timer.timeit(iters)/iters])
    atimes = np.array(times).T
    pl.semilogy(atimes[0],atimes[1],label='actual times (us)')
    w = np.where( atimes[0]%2 == 0)[0]
    pl.semilogy(atimes[0][w],atimes[1][w],'o',label='even N')
    pl.semilogy(atimes[0], 
            [fft_time_estimate(int(i))*1e6 for i in atimes[0]], 
             ':',label='estimate')
    pl.title(stmt+ ': usec')
    nfactors = [len(primefactors(n)) for n in atimes[0]]
    wprime = np.where(np.array(nfactors)==1)[0]
    pl.semilogy(atimes[0][wprime],atimes[1][wprime],'s',label='prime N')
    fftw3_times = get_fftw3_speed(atimes[0].astype(int))
    pl.semilogy(atimes[0], 1e6*fftw3_times.T[1], label='fftw3 plan(est)')
    pl.legend()
    pl.show()

#"""

