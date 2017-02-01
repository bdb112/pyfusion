""" A faster replacement for matplotlib specgram using float32 fftw3 
Real-Complex routines.
So far about 4x faster for no overlap, 3x faster with overlap
Doesn't include time from return to prompt until display is updated.
and times do not have SIMD enabled (fft is not the big cost)

Strategy: 
  1/ Use fftw3  -> factor of 2-3. (see multiprocessing)
  2/ Use Real-Complex 32 bit precision ffts. another factor of 2  (looks like psd already does this)
  3/ Take log, abs and do window on a composite array for speed, and avoid sqrt
     this is effective as chunks are small (128-1024) so numpy
     overhead per call of a few us can be important.
  Expect about 2.3us (E4300, simd, 1thread, and 1.5 lenny) for a float32 real 512 pt fft
   --> so for 2**20 array,  2000*2.3 = 5ms!
  However the pyfftw overhead of about 10us has in impact for small bins like 512.
  The more direct interface of the other fftw3 could help here, but you would
  need to do copies on input and output - these take 1.5us each for float32, 512 bytes, 
  so would be a nett win of 5us*512 =2.5ms for 512x512 case (15%) and 40ms for 2**22 (8192) = 13%

  ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'`` (default), ``'FFTW_PATIENT'``
  and ``'FFTW_EXHAUSTIVE'``.
Improvements:
Use the Advanced interface plan_many to get all short samples done all at once.
Hopefully threads are allocated one chunk at a time.

Timings for no overlap, E4300 (power: no diff) clf() means preceded by clf, hold=1, using original (no simd) ubuntu libraries.
          specgram         This
       512x512 512x8192  512x512 512x8192 512x32768
clf()    96     804       69       360  
hold=0   125    765       62       351  
hold=1   43     695       17.7     307      1486

Sun 21: 360ms for hold=0, 512*8192?  instead of 351  cpu is showing 10% when idle, and 86 ms in imshow - no gain by reboot
(don't forget to change hold= in the imshow line)

With FFTW fft instead of numpy replacement, now 150 for ffts, 198 total (hold=0)
(noverlap=512, NFFT=1024 - 426 (incl imshow) 336-360 before)

This compares with the raw fft time (noverlap=0) of 8k*2.3 = 18ms
Summary of times:
  window mult   inp=  F.execute allout=    log   imshow
    7ms  16ms   15ms   11ms       8ms      77ms   50ms  184 cf 197actual
See /usr/lib/pymodules/python2.7/matplotlib/mlab.py

Can test animate speed by 
im_obj.set_data(im)  # 50ms (std) 80ms 1200x800
im_obj.set_data(im[::5,::20]) # 30ms 60ms 1200x800 
time for ii in range(10):draw()
 
"""
import pyfftw
import numpy as np
import pylab as pl
import timeit as Tmt
from time import time as seconds
#from bdb_utils import process_cmd_line_args
from pyfusion.utils import process_cmd_line_args

pyfftw.interfaces.cache.enable()

def specgram(x, NFFT=256, Fs=2, Fc=0, detrend=None, window = pl.hanning, noverlap=128, cmap=None, hold=None, dtype = np.float32, threads=1, im_obj=None, interpolation='nearest', pylab_scaling=True, fast=True):
    """ 
Return and optionally plot the spectrogram of the data in x, using fftw3 to 
  optimise the speed.  Note that using fftw3, most of the time
  is spent elsewhere, so perhaps a cython implementation would be more efficient.
So far only coded for float32 and float64

Slight increase in speed if the previous image_object is given in the arg list.

Tests below on E4300, power on, 1 thread
  * 6 secs til image for specgram of 16M points, NFFT=512, noverlap=128  std win
  * 1.8 secs til first image fast=1, im_obj=im_obj (1 sec more for float64)
  * e.g. (note this is a wide dynamic range (17 decades of power) so there
       will be slight errors visible in the float32 version.

    >>> time figure(3);s=specgram(arange(2**24)**n,Fs=1e6,NFFT=512,im_obj=s[3],dtype=float64);n=(n+1)%3

    matplotlib specgram has good dyn range for arange2*24, but error of ~ -8 for random
    """
    siglen = len(x)
    if Fc != 0: print('strange Fc value: {0}'.format(Fc))
    
    if np.issubdtype(dtype, np.float32): 
        outdtype = np.complex64
    elif np.issubdtype(dtype, np.float64): 
        outdtype = np.complex128
    elif np.issubdtype(dtype, np.float128): 
        outdtype = np.complex256
    elif np.issubdtype(dtype, np.complex): 
        raise ValueError('Use numpy.specgram for your complex ({dt}) input data'
                        .format(dt=dtype))
    else: raise ValueError("specgram_fft23 can't use data type {dt}" .format(dt=dtype))

    winarr1 = window(NFFT).astype(np.dtype)

    if noverlap == 0:
        nbins = siglen/NFFT
        seglen = NFFT
        winarr_all = np.tile(winarr1, nbins) 
        windata32 = winarr_all*x  # tiling and bulk multiply saves 30ms/180ms

    else:
        seglen = NFFT - noverlap
        nbins = (siglen-noverlap)/seglen  # make sure noverlap left over at end
        windata32 = np.empty(NFFT*nbins, dtype = dtype)
        for b in range(nbins): 
            windata32[b*NFFT:(b+1)*NFFT] = \
                x[seglen*b:seglen*b+NFFT] * winarr1

    simd_align =  pyfftw.simd_alignment  # 16 at the moment.
    inp = pyfftw.n_byte_align_empty(NFFT, simd_align, dtype)
    out = pyfftw.n_byte_align_empty(NFFT/2+1,  simd_align, dtype = outdtype)
    allout = np.empty((nbins,NFFT/2+1),dtype=outdtype)
    F = pyfftw.FFTW(inp,out,threads=threads)#, force_nogil=force_nogil)

    for b in range(nbins):
        inp[:] = windata32[b*NFFT:(b+1)*NFFT]
        F.execute()
        allout[b,:] = out

    P = np.real(allout.T)**2 + np.imag(allout.T)**2

    P[[0,-1]] *=0.5
    Fs = float(Fs) # make sure!
    if pylab_scaling: P = P*2/Fs/(np.abs(winarr1)**2).sum()
    im = 10/np.log(10)*np.log(P)
    extent = (0, nbins*seglen/float(Fs),0,Fs/2)

    # first check a few things
    # in spite of this logic, you can fool it by putting a different image in
    # really should carefully find child 1 of figure, and check that the
    # newest image child of that has the same extent!
    fig=pl.gcf()
    if im_obj is not None:
        if not (np.allclose(extent, im_obj.get_extent()) and 
                im_obj.get_animated() and
                (im_obj.get_interpolation() == interpolation) and
                (im_obj.get_figure() == fig)):
            print('not using old object - mismatch with {ii} or not animated'
                  .format(ii=im_obj))
            im_obj=None

        try:  # is there an image that looks right?
            chs = fig.get_children()[1].get_children()
            is_img = [hasattr(ch,'write_png') for ch in chs]
            latest_image = chs[np.where(is_img)[0][-1]] # last image
            if not latest_image==im_obj: 
                im_obj=None
                print('image (im_obj) does not match image in figure')
        except:
            print('Failed to find similar image in figure')
            im_obj=None
            
    if im_obj is None:  # seems OK, lets do it.
        #im_obj = pl.imshow(im,origin='lower',aspect='auto',hold=hold)
        im_obj = pl.imshow(im,origin='lower',aspect='auto',animated=True, interpolation=interpolation, hold=hold, extent=extent)
        pl.show()
    else:
        sh = np.shape(im)
        pix = fig.get_size_inches()*fig.get_dpi()
        if fast and (np.prod(sh)>2e5):
            im_obj.set_data(im[::max(1,0.5*sh[0]/pix[0]),
                                 ::max(1,0.5*sh[1]/pix[1])]),
            pl.draw() 
            print('now full res')
        im_obj.set_data(im)
        pl.draw() 
    frange = np.linspace(extent[2], extent[3], np.shape(im)[0]) #agrees with pylabn
    t0range = np.linspace(extent[0], extent[1], np.shape(im)[1]+1)
    trange = t0range[0:-1]+NFFT/(2*Fs) # midway point
    return(P, frange, trange, im_obj)

def test_specgram(x=np.random.random(2**12), NFFT=256, noverlap=128, dtype='float32', rtol=1e-6, atol=3e-7, verbose=1):
    """  compare with pylab specgram for random noise
    atol=2e-7 shows errors < 1 in 10,000 times  - verbose not yet implemented
    don't run too many of these  (i.e. > 1000)- eats into RAM 
    """
    from numpy.testing import  assert_allclose, assert_almost_equal
    mpS = pl.specgram(x, NFFT=NFFT, noverlap=noverlap)
    w3S = specgram(x, NFFT=NFFT, noverlap=noverlap, dtype=dtype)
    for i in range(3):
        # note that assert_allclose defaults atol to 0, but allclose defaults 
        # atol to 1e-8
        assert_allclose(mpS[i], w3S[i],rtol=rtol, atol=atol)

def all_tests_specgram(slow=False):
    """ compare with pylab specgram for random noise data size=2**12 (and 
    2*22 for slow=True)
    """
    test_specgram()
    test_specgram(noverlap=0)
    test_specgram(dtype=np.float64)
    if slow: test_specgram(np.random.random(2**22),dtype=np.float64)

if __name__ == "__main__":
    test_specgram()

'''

_var_defaults = """
siglen = 2**18
NFFT = 512
noverlap = 0
winfn = pl.hanning
wait = 1
threads = 1
test_numpy = 0
"""
exec(_var_defaults)
exec(process_cmd_line_args())


# rough attempt to make it look interesting for a range of samples
testtime = np.linspace(0,siglen/2.**19,siglen)
(w1,w2) = 2*np.pi*np.array([2e3*(1+testtime**2),20e3 + testtime])  # parabolic  for LF, slight ramp for HF
testdata = np.sin(w1*testtime) + np.sin(w2*testtime) + np.random.random(siglen)
data32 = testdata.astype(np.float32)

if test_numpy:
    st = seconds()
    x= pl.psd(data32,NFFT=NFFT,noverlap=noverlap,window=winfn(NFFT))
    print(seconds()-st)
    pl.clf()

    if wait: raw_input('continue for np.specgram')  # to see consistent timing - im takes longer if there has been a clf


    # time numpy
    st = seconds()
    plspec = pl.specgram(testdata,NFFT=NFFT,noverlap=noverlap,window=winfn(NFFT),hold=0)
    print(seconds()-st, np.shape(plspec[0]))
    pl.show()
    #pl.figure()

    if wait: raw_input('continue to fftw3 version')


simd_align =  pyfftw.simd_alignment  # 16 at the moment.
window = winfn(NFFT).astype(np.float32)
# time pyfftw
st = seconds()
if noverlap == 0:
    nbins = siglen/NFFT
    winarr = np.tile(window, nbins) 
    windata32 = winarr*data32  # tiling and bulk multiply saves 30ms/180ms

else:
    seglen = NFFT - noverlap
    nbins = (siglen-noverlap)/seglen  # make sure noverlap left over at end
    windata32 = np.empty(NFFT*nbins, dtype = 'float32')
    for b in range(nbins): 
        windata32[b*NFFT:(b+1)*NFFT] = data32[seglen*b:seglen*b+NFFT] * window

inp = pyfftw.n_byte_align_empty(NFFT, simd_align, 'float32')
out = pyfftw.n_byte_align_empty(NFFT/2+1,  simd_align, dtype=np.complex64)
allout = np.empty((nbins,NFFT/2+1),dtype=np.complex64)
F = pyfftw.FFTW(inp,out,threads=threads)#, force_nogil=force_nogil)

for b in range(nbins):
    inp[:] = windata32[b*NFFT:(b+1)*NFFT]
    F.execute()
    allout[b,:] = out
    # out[b,:] = pyfftw.interfaces.numpy_fft.rfft(windata32[b*NFFT:(b+1)*NFFT])
    # 512:2**20 32bit: fft part is 47ms, rest is 19ms - quite big?

im = 10/np.log(10)*np.log(np.real(allout.T)**2 + np.imag(allout.T)**2)  # 35 ns per element (E4300)
#im = 20/np.log(10)*np.log(np.abs(out.T))  # 51 ns per element (E4300)
# 7ns for abs,40 for log10(float32)  25 for log() (in gcc 22ns) , 2 for 10*
b4im=seconds()-st
pl.imshow(im,origin='lower',aspect='auto',hold=1)
print('total time = {t:.1f} ms, before im = {b4:.1f}, np.shape={sh} '
      .format(t=1e3*(seconds() - st), b4=1e3*b4im, sh=np.shape(im)))  # 66 ms cf 231 ms - a little disappointing.
pl.show()
"""

time xx=pyfftw.interfaces.numpy_fft.rfft(windata32[b*NFFT:(b+1)*NFFT],planner_effort='FFTW_EXHAUSTIVE')

x=empty(2**19,dtype=float32)
%timeit for b in range(1023): x[b*512:b*512+512]=data32[256*b:256*b+512]
1000 loops, best of 3: 1.38 ms per loop

"""
'''
