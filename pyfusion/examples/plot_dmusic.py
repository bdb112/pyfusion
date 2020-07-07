from six.moves import input

import sys, warnings
from numpy import sqrt, argsort, average, mean, pi, array, fft, log10
import pyfusion as pf
import pyfusion.utils
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy  # for the cache
from pyfusion.data.dmusic import dmusic

try:
    import getch
    use_getch = True
except:
    use_getch = False
print(" getch is %savailable" % (['not ', ''][use_getch]))

_var_defaults="""

diag_name = ''
debug=0 #  > 0 stops in dmusic - special value -1 plots all the contours.
dev_name='W7X'   # 'LHD'
hold=0
exception=Exception  # to catch all exceptions and move on
decimate=50
chan=0
lowpass = None  # set to corner freq for lowpass filter
highpass = None  # set to corner freq for lowpass filter
time_range=[5.8,5.81]
wRg = []  # range of w in the interval [0-2pi]
dRg = []  # damping range
dt=1e-5
seg_time = 0.1  # length of data retrieved per run
J=50
K=4
numpts = 512
baselev=1e-4
normalise='0'
interpolation=plt.rcParams['image.interpolation']  #  (default) 'nearest','bilinear, bicubic
myfilter3=dict(passband=[2e3,4e3], stopband=[1e3,6e3], max_passband_loss=2, min_stopband_attenuation=15,btype='bandpass')
filter = None  
help=0
separate=1
verbose=0
shot_number = None
# to set freq range  pyfusion.config.set('Plots','FT_axis','[0.01, 0.07, 30, 50]')
stop=False  # True will stop to allow debugging
"""
exec(_var_defaults)
exec(pf.utils.process_cmd_line_args())
if help==1: 
    print(__doc__) 
    exit()

if dev_name == 'LHD': 
    if diag_name == '': diag_name= 'MP2010'
    if shot_number is None: shot_number = 27233
    #shot_range = range(90090, 90110)
elif 'H1' in dev_name:
    if diag_name == '': diag_name = "H1DTacqAxial"
    if shot_number is None: shot_number = 69270
elif 'W7' in dev_name:
    diag_name='W7X_MIRNOV_41_3' if diag_name == '' else diag_name
    shot_number=[20181009,24] if shot_number is None else shot_number
    
device = pf.getDevice(dev_name)

try:
    shot_cache
except:
    print('shot cache not available - use run -i next time to enable')
    shot_cache = {}


this_key = "{s}:{d}".format(s=shot_number, d=diag_name)
if this_key in shot_cache:  # we can expect the variables to be still around, run with -i
    d = deepcopy(shot_cache[this_key])
else:
    print('get data for {k}'.format(k=this_key))
    d = device.acq.getdata(shot_number, diag_name, time_range=time_range, contin=not stop)
    shot_cache.update({this_key: deepcopy(d)})

if time_range is not None:
    d = d.reduce_time(time_range, fftopt=True)  # could use d.reduce_time(copy=False,time_range)

if lowpass is not None: 
    if highpass is None:
        d = d.sp_filter_butterworth_bandpass(
            lowpass*1e3,lowpass*2e3,2,20,btype='lowpass')
    else:
        bp = [1e3*lowpass,1e3*highpass]
        bs = [0.5e3*lowpass,1.5e3*highpass]
        d = d.sp_filter_butterworth_bandpass(bp, bs,2,20,btype='bandpass')
elif filter is not None:
    if 'btype' in filter:
        d = d.sp_filter_butterworth_bandpass(**filter)
    else:
        (fc,df) = (filter['centre'],filter['bw']/2.)
        d = d.filter_fourier_bandpass(
            passband=[fc-df,fc+df], stopband = [fc-2*df, fc+2*df], 
            taper = filter['taper'])
else:
    pass # no filter

if decimate > 1:
    d.signal = d.signal[:, ::decimate] if len(np.shape(d.signal)) == 2 \
        else d.signal[::decimate]

    d.timebase = d.timebase[::decimate]

# Make a series of spectra into an image
segs = [d.reduce_time([t0, t0+dt], copy=True)
        for t0 in arange(time_range[0], time_range[1]-dt, dt)]  # should correct the time base
#  cut down to the shortest
if len(segs[-1].timebase) < len(segs[0].timebase):
    print('Discarded the last short segment')
    segs = segs[0:-1]
if len(segs) < 1:
    raise LookupError('Need at least one segment to analyse')
seg_len = min([len(seg.timebase) for seg in segs])
J = seg_len//2 if J==0 else J

imf = array([abs(fft.fft((seg.signal if chan is None else seg.signal[chan]))[0:seg_len])
            for seg in segs])  # drop the last - probably short
fig, [axf, axdm] = plt.subplots(2, 1)

dtsample = np.average(np.diff(seg.timebase))
frange = 1e-3 * array([0,1])/(dtsample)

axf.imshow(log10(imf.T + baselev), aspect='auto', origin='lower',
           extent=list(time_range) + list(frange))

wRg = linspace(0.1,0.5,40) if wRg == [] else wRg
if max(wRg) > 13:  # assume it is in Hz
    wRg = wRg * 2*pi * dtsample

dRg = linspace(-0.5,.5,20) if dRg == [] else dRg
# dm=dmusic(segs[0].signal[2], K=K, J=50, wRg=wRg, dRg=dRg,plot=1)
estimate = 4e-6 * len(wRg) * len(dRg) * J * len(segs)
print('Estimate {est:.1f} seconds'.format(est=estimate))

imdm = array([sum(dm, axis=0)
              for dm in [dmusic(seg.signal if chan is None else seg.signal[chan],
                                K=4, J=J, wRg=wRg, dRg=dRg, debug=debug, plot=debug==-1)
                         for seg in segs]])

frange =  array([min(wRg), max(wRg)])/(2*pi*dtsample)
axdm.imshow(log10(imdm.T), origin='lower', aspect='auto', interpolation=interpolation,
            extent=list(time_range) + list(frange * 1e-3))
axdm.set_ylabel('f(kHz)')
n_used = 2*J # what about K?
axdm.set_xlabel('t(sec): [sample is {J}/{n_used}=>{sl:.2g}s, {np:.1f} periods of the highest freq. scanned]'
                .format(sl=n_used * dtsample, np=frange[1] * n_used*dtsample,
                        n_used=n_used, J=J))
fig.suptitle('Shot {sh}: {di}, dt={dt}'.format(sh=str(shot_number), di=diag_name, dt=dt))
plt.show()

