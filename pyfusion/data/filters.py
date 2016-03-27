"""
Some un-pythonic code here (checking instance type inside
function). Need to figure out a better way to do this.

python3 issues:

/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/filters.py:65: DeprecationWarning: classic int division
  nice = [2**p * n/16 for p in range(minp2,maxp2) for n in [16, 18, 20, 24, 27]]
/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/utils.py:120: DeprecationWarning: classic int division
  ipks = find_peaks(np.abs(FT)[0:ns/2], minratio = minratio, debug=1)
/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/filters.py:443: DeprecationWarning: classic int division
  twid = 2*(1+max(n_pb_low - n_sb_low,n_sb_hi - n_pb_hi)/2)
/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/base.py:57: UserWarning: 
defaulting taper to 1 as band edges are sharp
/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/filters.py:508: DeprecationWarning: classic int division
  if np.mod(NA,2)==0: mask[:NA/2:-1] = mask[1:(NA/2)]   # even
/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/filters.py:490: DeprecationWarning: classic int division
  low_mid = n_pb_low - twid/2
/home/bdb112/pyfusion/mon121210/pyfusion/pyfusion/data/filters.py:491: DeprecationWarning: classic int division
  high_mid = n_pb_hi + twid/2

"""


from datetime import datetime
from pyfusion.debug_ import debug_
from pyfusion.utils.utils import warn
from copy import deepcopy
from numpy import searchsorted, arange, mean, resize, repeat, fft, conjugate, linalg, array, zeros_like, take, argmin, pi, cumsum
from numpy import correlate as numpy_correlate
import numpy as np
from time import time as seconds
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

try:
    from scipy import signal as sp_signal
except:
    # should send message to log...
    pass

import pyfusion

DEFAULT_SEGMENT_OVERLAP = 1.0


def cps(a,b):
    return fft.fft(a)*conjugate(fft.fft(b))  # bdb 30% fft


filter_reg = {}


def register(*class_names):
    def reg_item(filter_method):
        for cl_name in class_names:
            if cl_name not in filter_reg:
                filter_reg[cl_name] = [filter_method]
            else:
                filter_reg[cl_name].append(filter_method)
        return filter_method
    return reg_item

"""
class MetaFilter(type):
    def __new__(cls, name, bases, attrs):
        filter_methods = filter_reg.get(name, [])
        attrs.update((i.__name__,i) for i in filter_methods)
        return super(MetaFilter, cls).__new__(cls, name, bases, attrs)
"""

def next_nice_number(N):
    """ return the next highest power of 2 including nice fractions (e.g. 2**n *5/4)
    takes about 10us  - should rewrite more carefully to calculate
    starting from smallest power of 2 less than N, but hard to do better
    >>> print(next_nice_number(256), next_nice_number(257))
    (256, 288)

    Have to be careful this doen't take more time than it saves!
    """
    if N is None:
        (minp2, maxp2) = (6, 16)
    else:
        minp2 = int(np.log2(N))
        maxp2 = minp2 + 2

    nice = [2**p * n/16 for p in range(minp2, maxp2) for n in [16, 18, 20, 24, 27]]
    if N is None: return(np.array(nice))
    for n in nice:
        if n>=N: return(n)


def get_optimum_time_range(input_data, new_time_range):
    """ This grabs a few more (or a few less, if enough not available)
    points so that the FFT is more efficient.  For FFTW, it is more
    efficient to zero pad to a nice number above even if it is a long
    way away.  This is always true for Fourier filtering, in which
    case you never see the zeros.  For general applications, the zeros
    might be confusing if you forget they have been put there.
    """
    from pyfusion.utils.primefactors import fft_time_estimate

    nt_args = searchsorted(input_data.timebase, new_time_range)
    # try for 20 more points
    extension = ((new_time_range[1]-new_time_range[0])
                 * float(20)/(nt_args[1]-nt_args[0]))
    (dum, trial_upper) = searchsorted(input_data.timebase,
                                      [new_time_range[0],
                                       new_time_range[1]+extension])
    # if not consider using less than the original request
    trial_lower = trial_upper - 20
    times = []
    for num in range(trial_lower, trial_upper):
        times.append(fft_time_estimate(num - nt_args[0]))

    newupper = trial_lower+np.argmin(times)
    if newupper != nt_args[1]: 
        pyfusion.utils.warn('Interval fft optimized from {old} to {n} points'
                            .format(n=newupper-nt_args[0], 
                                    old=nt_args[1]-nt_args[0]))

    best_upper_time = input_data.timebase[newupper]
    new_time_range[1] = (best_upper_time
                         - 0.5*np.average(np.diff(input_data.timebase)))
    if pyfusion.VERBOSE > 0:
        print('returning new time range={n}'.format(n=new_time_range))
    return(new_time_range)


@register("TimeseriesData", "DataSet")
def reduce_time(input_data, new_time_range, fftopt=False):
    """ reduce the time range of the input data in place(copy=False)
    or the returned Dataset (copy=True - default at present).
    if fftopt, then extend time if possible, or if not reduce it so that
    ffts run reasonably fast. Should consider moving this to actual filters?
    But this way users can obtain optimum fft even without filters.
    The fftopt is only visited when it is a dataset, and this isn't happening
    """
    from pyfusion.data.base import DataSet
    if pyfusion.VERBOSE > 1:
        print('Entering reduce_time, fftopt={0}, isinst={1}'
              .format(fftopt, isinstance(input_data, DataSet)))
        pyfusion.logger.warning("Testing: can I see this?")
    if (min(input_data.timebase) >= new_time_range[0] and 
        max(input_data.timebase) <= new_time_range[1]):
        print('time range is already reduced')
        return(input_data)

    if isinstance(input_data, DataSet):
        if fftopt: new_time_range = get_optimum_time_range(input_data, new_time_range)

        # output_dataset = input_data.copy()
        # output_dataset.clear()
        print('****new time range={n}'.format(n=new_time_range))
        output_dataset = DataSet(input_data.label+'_reduce_time')
        for data in input_data:
            try:
                output_dataset.append(data.reduce_time(new_time_range))
            except AttributeError:
                pyfusion.logger.warning("Data filter 'reduce_time' not applied to item in dataset")
        return output_dataset
    # else: this is effectively a matching 'else' - omit to save indentation
    # ??? this should not need to be here - should only be called from
    # above when passed as a dataset (more efficient)
    if fftopt: new_time_range = get_optimum_time_range(input_data, new_time_range)
    new_time_args = searchsorted(input_data.timebase, new_time_range)
    input_data.timebase = input_data.timebase[new_time_args[0]:new_time_args[1]]
    if input_data.signal.ndim == 1:
        input_data.signal = input_data.signal[new_time_args[0]:new_time_args[1]]
    else:
        input_data.signal = input_data.signal[:, new_time_args[0]:new_time_args[1]]
    if pyfusion.VERBOSE>1: print('reduce_time to length {l}'
                                 .format(l=np.shape(input_data.signal))),
    return input_data


@register("TimeseriesData", "DataSet")
def segment(input_data, n_samples, overlap=DEFAULT_SEGMENT_OVERLAP):
    """Break into segments length n_samples.

    Overlap of 2.0 starts a new segment halfway into previous, overlap=1 is
    no overlap.  overlap should divide into n_samples.  Probably should
    consider a nicer definition such as in pyfusion 0
    """
    from .base import DataSet
    from .timeseries import TimeseriesData
    if n_samples<1:
        dt = np.average(np.diff(input_data.timebase))
        n_samples = next_nice_number(n_samples/dt)
        print('used {n} sample segments'.format(n=n_samples))

    if isinstance(input_data, DataSet):
        output_dataset = DataSet()
        for ii,data in enumerate(input_data):
            try:
                output_dataset.update(data.segment(n_samples))
            except AttributeError:
                pyfusion.logger.warning("Data filter 'segment' not applied to item in dataset")
        return output_dataset
    output_data = DataSet('segmented_%s, %d samples, %.3f overlap' %(datetime.now(), n_samples, overlap))
    # python3 check this
    for el in range(0,len(input_data.timebase), int(n_samples/overlap)):
##  was  for el in arange(0,len(input_data.timebase), n_samples/overlap):
        if input_data.signal.ndim == 1:
            tmp_data = TimeseriesData(timebase=input_data.timebase[el:el+n_samples],
                                      signal=input_data.signal[el:el+n_samples],
                                      channels=input_data.channels, bypass_length_check=True)
        else:
            tmp_data = TimeseriesData(timebase=input_data.timebase[el:el+n_samples],
                                      signal=input_data.signal[:,el:el+n_samples],
                                      channels=input_data.channels, bypass_length_check=True)
            
        tmp_data.meta = input_data.meta.copy()
        tmp_data.history = input_data.history  # bdb - may be redundant now meta is copied
        output_data.add(tmp_data)
    return output_data

@register("DataSet")
def remove_noncontiguous(input_dataset):
    remove_list = []
    for item in input_dataset:
        if not item.timebase.is_contiguous():
            remove_list.append(item)
    for item in remove_list:
        input_dataset.remove(item)
    return input_dataset

@register("TimeseriesData", "DataSet")
def normalise(input_data, method=None, separate=False):
    """ method=None -> default, method=0 -> DON'T normalise
    """
    from numpy import mean, sqrt, max, abs, var, atleast_2d
    from pyfusion.data.base import DataSet
    # this allows method='0'(or 0) to prevent normalisation for cleaner code
    # elsewhere
    if pyfusion.DEBUG>3: print('separate = %d' % (separate))
    if (method == 0) or (method == '0'): return(input_data)
    if (method is None) or (method.lower() == "none"): method='rms'
    if isinstance(input_data, DataSet):
        output_dataset = DataSet(input_data.label+"_normalise")
        for d in input_data:
            output_dataset.add(normalise(d, method=method, separate=separate))
        return output_dataset
    if method.lower() in ['rms', 'r']:
        if input_data.signal.ndim == 1:
            norm_value = sqrt(mean(input_data.signal**2))
        else:
            rms_vals = sqrt(mean(input_data.signal**2, axis=1))
            if separate == False:
                rms_vals = max(rms_vals)
            norm_value = atleast_2d(rms_vals).T            
    elif method.lower() in ['peak', 'p']:
        if input_data.signal.ndim == 1:
            norm_value = abs(input_data.signal).max(axis=0)
        else:
            max_vals = abs(input_data.signal).max(axis=1)
            if separate == False:
                max_vals = max(max_vals)
            norm_value = atleast_2d(max_vals).T
    elif method.lower() in ['var', 'variance', 'v']:
        if input_data.signal.ndim == 1:
            norm_value = var(input_data.signal)
        else:
            var_vals = var(input_data.signal, axis=1)
            if separate == False:
                var_vals = max(var_vals)
            norm_value = atleast_2d(var_vals).T            
    input_data.signal = input_data.signal / norm_value
    #print('norm_value = %s' % norm_value)
    norm_hist = ','.join(["{0:.2g}".format(float(v)) for v in norm_value.flatten()])
    input_data.history += "\n:: norm_value =[{0}]".format(norm_hist)
    input_data.history += ", method={0}, separate={1}".format(method, separate)
    input_data.scales = norm_value

    debug_(pyfusion.DEBUG, level=2, key='normalise',msg='about to return from normalise')
    return input_data
    
@register("TimeseriesData")
def svd(input_data):
    from .timeseries import SVDData
    svddata = SVDData(input_data.timebase, input_data.channels, linalg.svd(input_data.signal, 0))
    svddata.history = input_data.history
    svddata.scales = input_data.scales # need to pass it on to caller
    if pyfusion.DEBUG>4: print("input_data.scales",input_data.scales)
    debug_(pyfusion.DEBUG, level=2, key='svd',msg='about to return from svd')
    return svddata


#@register("TimeseriesData", "SVDData")
def fs_group_geometric(input_data, max_energy = 1.0):
    """
    no filtering implemented yet
    we don't register this as a filter, because it doesn't return a Data or DataSet subclass
    TODO: write docs for how to use max_energy - not obvious if using flucstruc() filter...
    """
    from .timeseries import SVDData
    #from base import OrderedDataSet

    if not isinstance(input_data, SVDData):
        input_data = input_data.subtract_mean().normalise(method="var").svd()

    output_fs_list = []#OrderedDataSet()

    if max_energy < 1.0:
        max_element = searchsorted(cumsum(input_data.p), max_energy)
        remaining_ids = range(max_element)
    else:
        remaining_ids = list(range(len(input_data.svs)))
    
    self_cps = input_data.self_cps()

    while len(remaining_ids) > 1:
        rsv0 = remaining_ids[0]
        tmp_cp = [mean(abs(cps(input_data.chronos[rsv0], input_data.chronos[sv])))**2/(self_cps[rsv0]*self_cps[sv]) for sv in remaining_ids]
        tmp_cp_argsort = array(tmp_cp).argsort()[::-1]
        sort_cp = take(tmp_cp,tmp_cp_argsort)
        delta_cp = sort_cp[1:]-sort_cp[:-1]
        
        output_fs_list.append([remaining_ids[i] for i in tmp_cp_argsort[:argmin(delta_cp)+1]])
            

        for i in output_fs_list[-1]: remaining_ids.remove(i)
    if len(remaining_ids) == 1:
        output_fs_list.append(remaining_ids)

    return output_fs_list


#@register("SVDData")
def fs_group_threshold(input_data, threshold=0.7):   # was 0.2 in earlier version
    """
    no filtering implemented yet
    we don't register this as a filter, because it doesn't return a Data or DataSet subclass
    """
    from timeseries import SVDData

    if not isinstance(input_data, SVDData):
        input_data = input_data.subtract_mean().normalise(method="var").svd()
    
    
    #svd_data = linalg.svd(norm_data.signal,0)
    output_fs_list = []

    #svs_norm_energy = array([i**2 for i in svd_data[1]])/input_data.E

    #max_element = searchsorted(cumsum(svs_norm_energy), energy_threshold)
    #remaining_ids = range(max_element)
    remaining_ids = list(range(len(input_data.svs)))
    
    self_cps = input_data.self_cps()

    while len(remaining_ids) > 1:
        rsv0 = remaining_ids[0]
        tmp_cp = [mean(abs(cps(input_data.chronos[rsv0], input_data.chronos[sv])))**2/(self_cps[rsv0]*self_cps[sv]) for sv in remaining_ids]
        filtered_elements = [i for [i,val] in enumerate(tmp_cp) if val > threshold]
        output_fs_list.append([remaining_ids[i] for i in filtered_elements])
            

        for i in output_fs_list[-1]: remaining_ids.remove(i)
    if len(remaining_ids) == 1:
        output_fs_list.append(remaining_ids)

    return output_fs_list

@register("TimeseriesData")
def flucstruc(input_data, min_dphase = -pi, group=fs_group_geometric, method='rms', separate=True, label=None, segment=0, segment_overlap=DEFAULT_SEGMENT_OVERLAP):
    """If segment is 0, then we dont segment the data (assume already done)"""
    from pyfusion.data.base import DataSet
    from pyfusion.data.timeseries import FlucStruc

    if label:
        fs_dataset = DataSet(label)
    else:
        fs_dataset = DataSet('flucstrucs_%s' %datetime.now())

    if segment > 0:
        for seg in input_data.segment(segment, overlap=segment_overlap):
            fs_dataset.update(seg.flucstruc(min_dphase=min_dphase, group=group, method=method, separate=separate, label=label, segment=0))
        return fs_dataset

    svd_data = input_data.subtract_mean().normalise(method, separate).svd()
    for fs_gr in group(svd_data):
        tmp = FlucStruc(svd_data, fs_gr, input_data.timebase, min_dphase=min_dphase, phase_pairs=input_data.__dict__.get("phase_pairs",None))
        tmp.meta = input_data.meta
        tmp.history = svd_data.history
        tmp.scales = svd_data.scales
        fs_dataset.add(tmp)    
    return fs_dataset


@register("TimeseriesData", "DataSet")
def subtract_mean(input_data):
    from pyfusion.data.base import DataSet
    if isinstance(input_data, DataSet):
        output_dataset = DataSet(input_data.label+"_subtract_mean")
        for d in input_data:
            output_dataset.add(subtract_mean(d))
        return output_dataset
    if input_data.signal.ndim == 1:
        mean_value = mean(input_data.signal)
        input_data.history += "\n:: mean_value\n%s" %(mean_value)
    else:
        mean_vector = mean(input_data.signal, axis=1)
        input_data.history += "\n:: mean_vector\n%s" %(mean_vector)
        mean_value = resize(repeat(mean_vector, input_data.signal.shape[1]), input_data.signal.shape)
    input_data.signal -= mean_value

    return input_data

###############################
## Wrappers to SciPy filters ##
###############################
@register("TimeseriesData")
def sp_filter_butterworth_bandpass(input_data, passband, stopband, max_passband_loss, min_stopband_attenuation,btype='bandpass'):
    """ 
      **   Warning - fails for a single signal in the enumerate step.
    This actually does ALL butterworth filters - just select bptype
    and use scalars instead of [x,y] for the passband.
     e.g df=data.sp_filter_butterworth_bandpass(2e3,4e3,2,20,btype='lowpass')
    """
    # The SciPy signal processing module uses normalised frequencies, so we need to normalise the input values
    norm_passband = input_data.timebase.normalise_freq(passband)
    norm_stopband = input_data.timebase.normalise_freq(stopband)
    ord,wn = sp_signal.filter_design.buttord(norm_passband, norm_stopband, max_passband_loss, min_stopband_attenuation)
    b, a = sp_signal.filter_design.butter(ord, wn, btype = btype)
    
    output_data = deepcopy(input_data)  # was output_data = input_data

    for i,s in enumerate(output_data.signal):
        if len(output_data.signal) == 1: print('bug for a single signal')
        output_data.signal[i] = sp_signal.lfilter(b,a,s)

    return output_data

def make_mask(NA, norm_passband, norm_stopband, input_data, taper):
    """  works well now, except that the stopband is adjusted to be
    symmetric about the passband (take the average of the differences
    The problem with crashes (zero mask) was solved by shifting the 
    mask before and after integrating, also a test for aliasing (on the
    mask before integration).
    """
    mask = np.zeros(NA)
    # define the 4 key points 
    #         /npblow-------------npbhi\
    # ___nsbl/                          \nsbhi____
    n_sb_low = int(norm_stopband[0]*NA/2)
    n_pb_low = int(norm_passband[0]*NA/2)
    n_pb_hi = int(norm_passband[1]*NA/2)
    n_sb_hi = int(norm_stopband[1]*NA/2)
    
    dt = float(np.average(np.diff(input_data.timebase)))
    if  n_sb_hi >= len(mask):
        raise ValueError('Filter frequency too high for data - units '
                         'problem? - sample spacing is {dt:.2g}'
                         .format(dt=dt))

    # twid is the transition width, and should default so that the sloped part is the same width as the flat?
    # !!! twid is not an input - !!!! doesn't do that yet.
    # make the transition width an even number, and the larger of the two
    # need to pull this code out and be sure it works.
    twid = 2*(1+max(n_pb_low - n_sb_low,n_sb_hi - n_pb_hi)//2)
    if (twid > (n_pb_low - n_sb_low)*3) or (twid > (n_sb_hi - n_pb_hi)*3):
        print('*********** Warning - unbalanced cutoff rate between high and low end'
              ' will cause the cutoff rates to be equalised widening one and reducing the other'
              ' difference between stop and pass bands should be similar ar both ends.')
    if (twid < 4):  # or (n_sb_low < 0):  #< not requ since fixed  
        if taper == 2: 
            raise ValueError(
            'taper 2 requires a bigger margin between stop and pass') 
        elif taper is None:
            warn('defaulting taper to 1 as band edges are sharp: twid={twid}'
                 .format(twid=twid))
            taper = 1
    else: 
        if taper is None:
            taper = 2

    if taper==1:
        #          _____
        #         /     \
        #        /       \
        # ______/         \___
        # want 0 at sb low and sb high, 1 at pb low and pb high
        # present code does not quite do this.
        # try to prevent zero width or very narrow (DC only) filters.
        if n_sb_low<0: n_sb_low=0
        if n_pb_low<0: n_pb_low=0
        if n_pb_hi<1: n_pb_hi=1
        if n_sb_hi<=n_pb_hi: n_sb_hi=n_pb_hi+1
        for n in range(n_sb_low,n_pb_low+1):
            if n_sb_low == n_pb_low:  # allow for pass=stop on low side
                mask[n]=1.
            else:
                mask[n] = float(n - n_sb_low)/(n_pb_low - n_sb_low) # trapezoid
        for n in range(n_pb_hi,n_sb_hi+1):
            mask[n] = float(n_sb_hi - n)/(n_sb_hi - n_pb_hi) # trapezoid
        for n in range(n_pb_low,n_pb_hi+1):
            mask[n] = 1
    elif taper == 2:
        # Note - must symmetrise (so that cumsum works)
        #          _
        #         / \
        #        |   |
        #  ______/   \___
        # want 0 at sb low and sb high, 1 at pb low and pb high
        # this means that the peak of the mask before integration is halfway between sb_low and pb_low
        # and pb_low - sb_low is an even number
        # present code does not quite do this.

        n_sb_low = n_pb_low-twid  # sacrifice the stop band, not the pass
        n_sb_hi = n_pb_hi+twid

        low_mid = n_pb_low - twid//2
        high_mid = n_pb_hi + twid//2
        for n in range(n_sb_low,low_mid):
            mask[n] = float(n - n_sb_low)/(low_mid - 1 - n_sb_low) # trapezoid
            mask[2*low_mid-n-1] = mask[n] #down ramp - repeat max
        #wid_up = n_sb_hi - n_pb_hi
        for n in range(n_pb_hi,high_mid): # negative tri
            mask[n] = float(n_pb_hi - n)/(high_mid - n_pb_hi - 1) # trapezoid
            mask[2*high_mid - n - 1] = mask[n]
        before_integration = mask    
        # after running filters.py, this should be OK   
        # make_mask(512, [0.8,.93], [0.9,.98],dat,2)
        # but changing 0.98 to 0.99 will give aliasing error. 
        if np.max(np.abs(mask[NA//2-4:NA//2+4]))>0:
            raise ValueError('mask aliasing error')
        # note: ifftshift is only different for an odd data length
        # the fftshifts were necessary to avoid weirdness if the
        # stopband went below zero freq.
        mask = np.fft.ifftshift(np.cumsum(np.fft.fftshift(mask))) # integrate
        if pyfusion.DEBUG>1:
            nonr = 0.5/dt
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(np.arange(len(mask))/dt/float(NA), mask, '.-')
            ax1.plot(np.arange(len(mask))/dt/float(NA), before_integration)
            ax1.set_xlabel('real freq. units, (norm on top scale), npoints={NA}, norm/real = {nonr}'
                           .format(NA=NA, nonr=nonr))
            ax2 = ax1.twiny()
            # this is my hack - it should be OK, but may be a little out
            ax2.set_xlim(np.array(ax1.get_xlim())/nonr)
            fig.suptitle('mask before normalisation - twid={twid}'.format(twid=twid))
            plt.show(0)

        if np.max(mask) == 0:
            raise ValueError('zero mask, '
                             'norm_passband = {pb}, norm_stopband={sb}, taper {t}'
                             .format(pb = norm_passband, sb = norm_stopband, t=taper))
        mask = mask/np.max(mask)
    # reflection only required for complex data
    # this even and odd is not totally thought through...but it seems OK
    if np.mod(NA,2)==0: mask[:NA/2:-1] = mask[1:(NA/2)]   # even
    else:            mask[:1+NA/2:-1] = mask[1:(NA/2)] # odd 
    return(mask)

@register("TimeseriesData")
def filter_fourier_bandpass(input_data, passband, stopband, taper=None, debug=None):
    """ 
    Note: Is MUCH (2.2x faster) more efficient to use real ffts, (implemented April)
    Use a Fourier space taper/tophat or pseudo gaussian filter to perform 
    narrowband filtering (much narrower than butterworth).  
    Problem is that bursts may generate ringing. 
    This should be better with taper=2, but it is not clear
    
    See the __main__ code below for nice test facilities
    twid is the width of the transition from stop to pass (not impl.?)
    >>> tb = timebase(np.linspace(0,20,512))
    >>> w = 2*np.pi* 1  # 1 Hertz
    >>> dat = dummysig(tb,np.sin(w*tb)*(tb<np.max(tb)/3))
    >>> fop = filter_fourier_bandpass(dat,[0.9,1.1],[0.8,1.2],debug=1).signal[0]

    Testing can be done on the dummy data set generated after running 
    filters.py
    e.g. (with pyfusion,DEBUG=2
    make_mask(512, [0.8,.93], [0.9,.98],dat,2)
    # medium sharp shoulder
    fopmed = filter_fourier_bandpass(dat,[9.5,10.5],[9,11],debug=1).signal[0]
    # very sharp shoulders
    fopsharp = filter_fourier_bandpass(dat,[9.1,10.9],[9,11],debug=1)
    """
    if debug is None: debug = pyfusion.DEBUG
# normalising makes it easier to think about - also for But'w'h 
    if (passband[0]<stopband[0]) or (passband[1]>stopband[1]):
        raise ValueError('passband {pb} outside stopband {sb}'
                         .format(pb=passband, sb=stopband))
    norm_passband = input_data.timebase.normalise_freq(np.array(passband))
    norm_stopband = input_data.timebase.normalise_freq(np.array(stopband))
    NS = len(input_data.timebase)
    NA = next_nice_number(NS)
    input_data.history += str(" {fftt} : nice number: {NA} cf {NS}\n"
                              .format(fftt=pyfusion.fft_type, NA=NA, NS=NS))
    # take a little more to speed up FFT

    mask = make_mask(NA, norm_passband, norm_stopband, input_data, taper)
    output_data = deepcopy(input_data)  # was output_data = input_data

    if (pyfusion.fft_type == 'fftw3'):
        # should migrate elsewhere, but the import is only 6us
        # the setup time seems about 150-250us even if size is in wisdom
        # for 384, numpy is 20us, fftw3 is 4us, so fftw3 slower for less than 
        # 10 channels (unless we cache the plan)
        #time# st=seconds()
        import pyfftw
        #time# im=seconds()
        tdtype = np.float32
        fdtype = np.complex64
        # this could be useful to cache.
        simd_align =  pyfftw.simd_alignment  # 16 at the moment.
        tdom = pyfftw.n_byte_align(np.zeros(NA,dtype=tdtype), simd_align)
        FT = pyfftw.n_byte_align_empty(NA/2+1, simd_align, fdtype)
        ids = [[id(tdom),id(FT)]]  # check to see if it moves out of alignment
        #time# alloc_t = seconds()
        fwd = pyfftw.FFTW(tdom, FT, direction='FFTW_FORWARD',
                          **pyfusion.fftw3_args)
        rev = pyfftw.FFTW(FT, tdom, direction='FFTW_BACKWARD',
                          **pyfusion.fftw3_args)
        #time# pl=seconds()
        #time# print("import {im:.2g}, alloc {al:.2g}, planboth {pl:.2g}"
        #time#      .format(im=im-st, al=alloc_t-im, pl=pl-alloc_t))
    else:
        tdtype = np.float32
        tdom = np.zeros(NA,dtype=tdtype)

        # example of tuning
        #pyfusion.fftw3_args= {'planning_timelimit': 50.0, 'threads':1, 'flags':['FFTW_MEASURE']}

    singl = not isinstance(output_data.signal, (list, tuple))
    if singl:
        output_data.signal = [output_data.signal]
                        
    for i,s in enumerate(output_data.signal):
        #if len(output_data.signal) == 1: print('bug for a single signal')

        #time run -i  pyfusion/examples/plot_svd.py "dev_name='LHD'" start_time=.497 "normalise='r'" shot_number=90091 numpts=512 diag_name=MP2010HMPno612 "filter=dict(centre=8e3,bw=5e3,taper=2)" plot_mag=1 plot_phase=1 separate=1 closed=0 time_range=[0.0000,4.]
        # 4.5 cf 15.8diag_name=MP2010HMPno612, time_range=[0.0000,2.80000] 
        # 0, 4.194304 2**21 samples, 21.8 cf 6.8 1thr
        # (0,2)secs 90091 =2000000 samples 17 np, 5.73 2thread, nosimd, 6.1 1thread (mem bw?) 3.2 sec no filt
        # note - the above are on an intermeittently loaded E4300 2 processor, below on 4 core 5/760
        # 0, 4.194304 2**21 samples, 10.9 cf 3.16 thr2 3.47 1thr and 2.0 secs no filter
        # for 17 fft/ifft takes about 1.16 sec 2 threads - should be (27.5ms+28.6)*17 = 952ms (14.2 2thr) OK
        # duplicate the fft execute lines  4.3(3.47)  2thr 3.7(3.16) extra 810ms (expect 14.2ms * 2 * 17) =482
        # the difference between 2 and 1thr should be 14*2*17 ms 500ms.
        # orignall - 90ms/channel extra in reverse trasnform - maybe the 50 sec limit stopped optimization
        # next _nice: 5.74 for 10 sec lenny 
        #  E4300: 1thr  9.3 (39np) for 10 sec 90091;    5.5 for 4 sec (19.6 np)
        if (pyfusion.fft_type == 'fftw3'):  # fftw3 nosim, no thread 2.8s cf 10s
            #time# sst = seconds()
            tdom[0:NS]=s  # indexed to make sure tdom is in the right part of memory
            if NS != NA: tdom[NS:]=0.
            fwd.execute()
            FT[:] = FT * mask[0:NA/2+1] # 12ms
            rev.execute()
            output_data.signal[i] = tdom[0:NS]/NA # doco says NA
            ids.append([id(tdom),id(FT)])
            #time# print("{dt:.1f}us".format(dt=(seconds()-sst)/1e-6)),
        else: # default to numpy
            tdom[0:NS] = s
            FT = np.fft.fft(tdom)
            IFT = np.fft.ifft(mask*FT)
            if np.max(np.abs(IFT.imag)) > 1e-6*np.max(np.abs(IFT.real)):
                pyfusion.logger.warning("inverse fft imag part > 1e-6")

            output_data.signal[i] = IFT.real[0:NS]
        
    if debug>2: print('ids of fftw3 input and output: {t}'.format(t=ids))
    if debug>1: 
        fig = plt.figure()
        #fplot = host_subplot(111)
        fplot = fig.add_subplot(111)
        tplot = fplot.twinx()
        tplot.plot(input_data.signal[0],'c',label='input')
        # for a while I needed a factor of 3 here too for fftw - why ????
        #plt.plot(output_data.signal[0]/(3*NA),'m',label='output/{N}'.format(N=3*NA))
        tplot.plot(output_data.signal[0],'m',label='output')
        tplot.set_ylim(-2.4,1.1)
        fplot.plot(mask,'r.-',label='mask')
        fplot.plot(np.abs(FT)/len(mask),label='FT')
        #fplot.set_ylim(-.2,3.8)
        #fplot.set_yscale('log', subsy=[2,5])
        #fplot.set_ylim(1e-7,1e5)
        fplot.set_yscale('symlog', linthreshy=1e-6)
        fplot.set_ylim(0,1e8)
        fig.suptitle('Passband {pbl}...{pbh}'
                        .format(pbl=passband[0],pbh=passband[1]))
        fplot.legend(loc=4)   # bottom right
        tplot.legend(loc=0)
        plt.show()
    debug_(debug, 2, key='filter_fourier')
    if np.max(mask) == 0: raise ValueError('Filter blocks all signals')
    if singl:
        output_data.signal = output_data.signal[0]

    return output_data


#########################################
## wrappers to numpy signal processing ##
#########################################
@register("TimeseriesData")
def downsample(input_data, skip=10, chan=None, copy=False):
    """ Good example of filter that changes the size of the data.
    """
    from .base import DataSet
    from .timeseries import TimeseriesData
    if isinstance(input_data, DataSet):
        output_dataset = DataSet()
        for ii,data in enumerate(input_data):
            try:
                output_dataset.update(data.downsample(skip))
            except AttributeError:
                pyfusion.logger.warning("Data filter 'downsample' not applied to item in dataset")
        return output_dataset
    # python3 check this
    if input_data.signal.ndim == 1:
        tmp_data = TimeseriesData(timebase=input_data.timebase[::skip],
                                  signal=input_data.signal[::skip],
                                  channels=input_data.channels, bypass_length_check=True)
    else:
        tmp_data = TimeseriesData(timebase=input_data.timebase[::skip],
                                  signal=input_data.signal[:,::skip],
                                  channels=input_data.channels, bypass_length_check=True)

    tmp_data.meta = input_data.meta.copy()
    tmp_data.history = input_data.history  # bdb - may be redundant now meta is copied
    return tmp_data

@register("TimeseriesData")
def integrate(input_data, baseline=[], chan=None, copy=False):
    """ Return the time integral of a signal, 
    Perhaps the first sample will be a Nan - no,the samples are all displaced by one
    If we used the trapeziodal method, the first and last samples are 'incorrect' 
    and Nans may be appropriate
    """
    # for now, always copy, too many problems otherwise.
    input_data = deepcopy(input_data)
    if copy: 
        input_data = input_data.copy()
    if (len(np.shape(input_data.signal)) == 2) and chan is None:
        for (s,sig) in enumerate(input_data.signal):
            input_data.signal[s][:] = integrate(input_data, chan=s, baseline=baseline).signal

    else:
        if chan is None:
            signal = input_data.signal
        else:
            signal = input_data.signal[chan]

        signal[:] = np.cumsum(signal)
        #signal[0] = np.nan   cumsum defines the first point

        input_data.signal = signal*np.average(np.diff(input_data.timebase))

    if baseline is not None:
        return remove_baseline(input_data, baseline, copy=False)
    else:
        return(input_data)

@register("TimeseriesData")
def remove_baseline(input_data, baseline=None, chan=None, copy=False):
    """ Remove a tilted baseline from a signal
    if the baseline is 4 elements, correct at two points (mid point of those intervals)
    baseline in the same units as the timebase
    """
    from pyfusion.data.convenience import whr, btw
    # for now, always copy, too many problems otherwise.
    input_data = deepcopy(input_data)

    if (len(np.shape(input_data.signal)) == 2) and chan is None:
        for (s,sig) in enumerate(input_data.signal):
            input_data.signal[s][:] = remove_baseline(input_data, chan=s, baseline=baseline).signal

    else:
        if chan is None:
            signal = input_data.signal
        else:
            signal = input_data.signal[chan]

        tb = input_data.timebase

        bl = np.array(baseline).flatten()
        if baseline is None or len(baseline) == 0:
            delta = .01
            bl = [np.min(tb), np.min(tb) + delta,
                        np.max(tb) - delta, np.max(tb)]

        wst = np.where(btw(tb, bl[0:2]))[0]
        wend = np.where(btw(tb, bl[2:4]))[0]

        bl_st = np.average(signal[wst])
        bl_end = np.average(signal[wend])

        time_st = np.average(tb[wst])
        time_end = np.average(tb[wend])
        print(bl_st, bl_end, time_st, time_end)

        input_data.signal = signal - (bl_st  * (time_end - tb)/(time_end - time_st)
                                      - bl_end * (time_st - tb)/(time_end - time_st))
        

    return(input_data)


@register("TimeseriesData")
def correlate(input_data, index_1, index_2, **kwargs):
    return numpy_correlate(input_data.signal[index_1],
                           input_data.signal[index_2], **kwargs)

if __name__ == "__main__":
# this is a pain - I can see the benefit of unit tests/nose tests. bdb
# make a class that looks like a timebase
    class timebase(np.ndarray):
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls).copy()
            return(obj)

        def normalise_freq(self, freq):
            return(2*freq*np.average(np.diff(self)))
        


    class dummysig():
        def __init__(self, tb,sig):
            """ timebase is given as an array in seconds
            30 (10 lots of three) signals are generated: as scaled copies 1,2,3x
            """
            self.timebase = tb
            self.signal = 10*[sig, 2*sig, 3*sig]
            self.history = ''

    import doctest
    from mpl_toolkits.axes_grid1 import host_subplot
    import matplotlib.pyplot as plt


    tb = timebase(np.linspace(0,20,2048))
    w = 2*np.pi* 10  # 10 Hertz
    wL = 2*np.pi* 0.1  # 10 Hertz
    dat = dummysig(tb,np.sin(w*tb) + np.sin(wL*tb)*(tb<np.max(tb)/3))
    # below is 680us/loop fftw3, 3x2k signals. 1.2ms with numpy
    #         1.84us                           8.5ms for 10*3 signals
    fop = filter_fourier_bandpass(dat,[9,11],[8,12],debug=1,taper=2).signal[0]
    plt.title('test 2 - 10Hz, +/- 1Hz'); plt.show()
    fopwide = filter_fourier_bandpass(dat,[8,12], [0,30],debug=1).signal[0]
    plt.title('test 2 - 10Hz wide'); plt.show()

    doctest.testmod()
