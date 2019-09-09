# from matlab exchange dmusic.m
#  see https://numpy.org/devdocs/user/numpy-for-matlab-users.html
import numpy as np
from numpy import pi, exp, log10, real, imag, conj, complex, arange, \
    array, shape, abs, zeros, matrix, linspace  
from numpy.linalg import svd, norm
from matplotlib import pyplot as plt

try:
    from pyfusion.debug_ import debug_
except ImportError:
    print("Try boyd's copy")
    from pyfusion.debug_ import debug_

debug = 0

def awgn(signal1, SNR_dB=40):
    import scipy as sp
    """ rough approx to matlab awgn
    Add white Gaussian noise to signal
    """

    # Desired linear SNR
    snr = 10.0**(SNR_dB/10.0)
    print "Linear snr = ", snr

    # Measure power of signal
    p1 = signal1.var()
    print "Power of signal1 = ", p1

    # Calculate required noise power for desired SNR
    n = p1/snr
    print "Desired noise power = ", n
    # doesn't make sense print "Calculated SNR =  %f dB" % (10*sp.log10(p1/n))

    # Generate noise with calculated power
    w = sp.sqrt(n)*sp.randn(len(signal1))

    # Add noise to signal
    s1 = signal1 + w
    return(s1)


#  maybereal = real # Use this to force reals to make checking easier.
def maybereal(val):
    return(val)
maybereal = real # Use this to force reals to make checking easier.

def dmusic(y, K, J, wRg=None, dRg=None, plot=None, debug=debug):
# function dmusic(y,K,J)
# 
# 1D Damped MUSIC (DMUSIC) Algorithm
#
# y = signal
# K = # of damped sinusoids
# J = Subsample of prediction matrix
# 
# Ex:
#    dmusic(y,2,12);
# 
# References:
#     [1] Y. Li, J. Razavilar, and K. Liu, "A Super-Resolution Parameter
#         Estimation Algorithm for Multi- Dimensional NMR Spectroscopy," 
#         University of Maryland, College Park, MD 1995.
#
#     [2] E. Yilmaz and E. Dilaveroglu, "A Performance Analysis for DMUSIC,"
#         presented at 5TH International Conference on Electrical and
#         Electronics Engineering (ELECO 2007), Bursa, Turkey, 2007.
#
# Coded by: Kenneth John Faller II January 07, 2008
#
#    warning('off','all');
#    
    y = maybereal(y)  #  can use just the real part if necessary
    N = len(y)-1;
    
    if((N-J) < K or J < K):
        raise ValueError('J has to be between K [{K}] and N-K [{NmK}]'
                         .format(K=K, NmK=N-K))
    #end
    
    # % Prediction Matrix
    # H = hankel(y,1:J);
    # A = H(1:J,:);
    A = array([y[i:i + J] for i in range(J)])
    if i + J -1 > N:
        raise LookupError('Too few samples for J={J}, (need {ns})'.format(ns=i+J, J=J))
    
    # % Singular Value Decomposition
    [U, D, V] = svd(A);    
    V = matrix(V).H  # to make it look as per matlab (using ndarrays)
    # Vn = V[:,K+1:J];
    Vn = V[:,K:J];

    # % MUSIC Spectrum    
    if wRg is not None:
        nW = len(wRg)
    else:
        nW = 63
        wRg = linspace(0, 6.2, nW)
    if dRg is not None:
        nB = len(dRg)
    else:
        nB = 11
        dRg = linspace(0, 1, nB)

    if wRg[-1] < 1/(2*pi):
        print('Warning - entire range is sub radian [{wr:.1f}]'.format(wr=wRg[-1])),

    s = zeros([nB,nW], dtype=complex);
    r = matrix(zeros([J,1], dtype=complex))
    f = 0 * s;  # doesn't need to be complex, doesn't hurt though
    
    for indB, B in enumerate(dRg):  #0:0.05:1
        for indW, w in enumerate(wRg): # =0:0.01:2*pi
            s[indB,indW] = -B + 1j*w;
            for indR in range(J)[0:0]:  #=0:J-1
                #n=indR+1;
                n = indR
                r[n,0] = maybereal(exp(indR*s[indB,indW]))
            # leave the loop in place but with nul executions so we can compare later
            rr = maybereal(exp(arange(J)*s[indB,indW]))
            r = np.matrix(np.stack([rr])).T
            rt = r/norm(r);  
            # % the prime ' is comp. cong. transpose (.H for *matrices* in py)
            # % the dot prime .' is transpose without conj (.T in py)
            # f(indB,indW) = real(1/((rt')*(conj(Vn)*(Vn.'))*rt));
            f[indB,indW] = real(1/((rt.H)*(conj(Vn)*(Vn.T))*rt));

    if plot is None and len(f.flatten()) > 2e5 or plot == False:
        if plot is None:
            print('suppressing plot - set plot=True')
    else:
        fig, [ax1, ax2] = plt.subplots(2, 1)
        fig.suptitle('K={K}, J, N={J}, {N}: '.format(**locals()) + ['complex','Re()'][maybereal.func_name=='real'])
        fmax = np.max(real(f)) # to set a suitable log range.
        ax1.contour(imag(s), real(s), real(f), 20)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Damping Factor')
        ax2.imshow( log10(real(f) + fmax/1e4), aspect='auto')
        ax2.set_xlabel('log10 of Re(f)')
        plt.show()
        plt.sca(ax1)  # to allow access to overplot after call
        
    debug_(debug, 1)
    return(real(f))
#$ =============================    
# function dmusicEx1()
# function dmusicEx1()
# 
# 1D Damped MUSIC (DMUSIC) Algorithm Example
# 
# This is the implementation of example 1 from reference [1]
# 
# Ex:
#    dmusicEx1();
# 
# References:
#     [1] Y. Li, J. Razavilar, and K. Liu, "A Super-Resolution Parameter
#         Estimation Algorithm for Multi- Dimensional NMR Spectroscopy," 
#         University of Maryland, College Park, MD 1995.
#
#     [2] E. Yilmaz and E. Dilaveroglu, "A Performance Analysis for DMUSIC,"
#         presented at 5TH International Conference on Electrical and
#         Electronics Engineering (ELECO 2007), Bursa, Turkey, 2007.
#
# Coded by: Kenneth John Faller II January 07, 2008
#

# from awgn import awgn


if __name__ == '__main__':
    test = "cfMatlab" # cfMatlab, LongSignal
    # probably OK to change this here, but change it at the top for safety
    # maybereal = real # Use this to force reals to make checking easier.

    j = complex(0, 1)
    if maybereal.func_name == 'real':
        print('** Testing real part of data only **\n')

    theta = 0.1;
    s1 = -0.2+j*2*pi*0.42;
    s2 = -0.1+j*2*pi*(0.42+theta);

    if test == 'cfMatlab':
        N = 8
        J = 4
        K = 2
        SNR = None
    elif test == 'LongSignal':
        N = 100
        J = 50
        K = 4
        SNR = 20
        s1 = -0.02+j*2*pi*0.05;
        s2 = -0.01+j*2*pi*(0.07);
    else:
        print('The full original example in dmusicEx1')
        N = 24
        J = 12
        K = 4
        SNR = 40
    
    """
    y = 0;
    for i in range(1, N+1):   # =1:N+1
        n=i-1;
        y[i] = exp(s1*n) + exp(s2*n);

    """
    n = arange(0, N+1) # the matlab version may be out by 1
    print('s1 = ', str(s1))
    if test == 'cfMatlab':
        y = exp(s1*n) # just one signal
        print('Just one signal')
    else:
        y = exp(s1*n) + exp(s2*n)        
        print('s2 = ', str(s2))

    if SNR is not None:  # make noise free to help comparison with matlab
        y = awgn(y,SNR);
    
    dmusic(y, K, J);
    axo = plt.gca()
    axo.plot(2*[imag(s1)], axo.get_ylim(),'gray', lw=0.5)
    axo.plot(axo.get_xlim(), 2*[real(s1)],'gray', lw=0.5)
    if 'cfMatlab' not in test:
        axo.plot(2*[imag(s2)], axo.get_ylim(),'r', lw=0.5)
        axo.plot(axo.get_xlim(), 2*[real(s2)],'r', lw=0.5)
"""
500x500 space with 12/24 samples, K=4 is about 10 secs
"""
