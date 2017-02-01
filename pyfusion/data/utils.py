import os, string
import random as _random
from numpy import fft, conjugate, array, mean, arange, searchsorted, argsort, pi
from pyfusion.utils.utils import warn
from pyfusion.debug_ import debug_
import pyfusion

import numpy as np

try:
    import uuid
except: # python 2.4
    pass

## for python <=2.5 compat, bin() is only python >= 2.6
## code taken from http://stackoverflow.com/questions/1993834/how-change-int-to-binary-on-python-2-5
def __bin(value):
    binmap = {'0':'0000', '1':'0001', '2':'0010', '3':'0011',
              '4':'0100', '5':'0101', '6':'0110', '7':'0111',
              '8':'1000', '9':'1001', 'a':'1010', 'b':'1011',
              'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111'}
    if value == 0:
        return '0b0'
    
    return '0b'+''.join(binmap[x] for x in ('%x' % (value,))).lstrip('0') or '0'

try:
    _bin = bin
except NameError: # python <= 2.5
    _bin = __bin

def unique_id():
    try:
        return str(uuid.uuid4())
    except:
        return ''.join(_random.choice(string.letters) for i in range(50))

def get_axes_pixcells(ax):
    """ return the  pixcell coorindates of the axes ax
    This is useful in determining how many characters will fit
    """
    return(ax.get_window_extent().bounds)

def cps(a,b):
    return fft.fft(a)*conjugate(fft.fft(b))    # bdb fft 10%

def subdivide_interval(pts, overlap= None, debug=0):
    """ return several intervals which straddle pts
    with overlap of ov
    The lowest x value is special - the point of division is much closer
    to that x then zero.
    overlap is a tuple (minoverlap, max), and describes the total overlap 
    """
    if overlap is None: overlap = [np.max(pts)/50, np.max(pts)/20]
    if len(overlap) == 1:
        warn('overlap should have a min and a max')
        overlap = [overlap/3.0, overlap]

    if (np.diff(pts)<0).any(): 
        warn('points out of order - reordering')
    pts = np.sort(pts)
    begins = []
    ends = []
    for (i, x) in enumerate(pts):
        if i == 0:
            divider = x * 0.8 - overlap[1]/2.
        else:
            divider = (x + pts[i-1])/2.

        if i == 0: 
            begins.append(0)
            ends.append(divider + overlap[1]/2.)
        else:
            this_overlap = min(max((divider - last_divider)/20,overlap[0]), 
                               overlap[1])/2.
            begins.append(last_divider - this_overlap)
            ends.append(divider + this_overlap)
        last_divider = divider

    if debug>1: print(begins, pts, ends)
    if debug>2: 
        import pylab as pl
        pl.figure()
        for i in range(len(pts)):
            pl.plot([begins[i],ends[i]],[i,i])
            if i>0: pl.plot([pts[i-1],pts[i-1]],[i,i],'o')
            pl.ylim(-1,20)
            pl.show()
    return(begins, ends)

def find_peaks(arr, minratio=.001, debug=0):
    """ find the peaks in the data in arr, by selecting points
    See also Shauns using find_peaks running average
    where the slope changes sign, and the value is > minratio*max(arr) """
    darr = np.diff(arr)
    wnz = np.where(darr != 0)[0]
    w_ch_sign = np.where(darr[wnz][0:-1]*darr[wnz][1:] < 0)[0]
    # now check these to find the max
    maxarr = np.max(arr[1:])  # to avoid zero freq
    maxi = []

    for i in w_ch_sign:
        darr_left = darr[wnz[i]]
        darr_right = darr[wnz[i]+1]
        if darr_left > 0:  # have a maximum
            imax = np.argmax(arr[wnz[i]:wnz[i]+2])
            iarrmax = wnz[i]+imax  # imax was relative to subarray
            if arr[iarrmax] > minratio*maxarr:
                maxi.append(iarrmax)
                if debug>1: print('arr elt {ii} = {v:.2f}'
                                  .format(ii=iarrmax, 
                                          v=arr[iarrmax]))
    debug_(pyfusion.DEBUG,level=2, key='find_peaks')
    return(np.array(maxi))


def find_signal_spectral_peaks(timebase, signal, minratio = .001, debug=0):
    ns = len(signal)
    FT = np.fft.fft(signal-np.average(signal))/ns  # bdb 0% fft?
    ipks = find_peaks(np.abs(FT)[0:ns/2], minratio = minratio, debug=1)
    fpks = ipks/np.average(np.diff(timebase))/float(ns)
    if debug>1:
        import pylab as pl
        pl.semilogy(np.abs(FT))
        pl.semilogy(ipks,np.abs(FT)[ipks],'o')

    return(ipks, fpks, np.abs(FT)[ipks])


def peak_freq(signal,timebase,minfreq=0,maxfreq=1.e18):
    """
    TODO: old code: needs review - since then bdb helped a bit...
    now also returns peaking factor
    this function only has a basic unittest to make sure it returns
    the correct freq in a simple case.

    >>> tb = np.linspace(0,1,10000)
    >>> int(peak_freq(np.sin(2*np.pi*567*tb), tb)[1])
    567

    """
    timebase = array(timebase)
    # Note - the call to this uses the first Svector
    sig_fft = fft.rfft(signal)    # bdb 5% fft (before rfft - was just fft)
    #sample_time = float(mean(timebase[1:]-timebase[:-1]))
    sample_time = np.average(np.diff(timebase))
    #fft_freqs = (1./sample_time)*arange(len(sig_fft)).astype(float)/(len(sig_fft)-1)
    # I think the -1 in (len() - 1) was an error - bdb
    fft_freqs = (1./sample_time)*arange(len(sig_fft)).astype(float)/len(signal)
    """ not needed for rfft # only show up to nyquist freq
    new_len = len(sig_fft)/2
    sig_fft = sig_fft[:new_len]
    fft_freqs = fft_freqs[:new_len]
    """
    [minfreq_elmt,maxfreq_elmt] = searchsorted(fft_freqs,[minfreq,maxfreq])
    sig_fft = sig_fft[minfreq_elmt:maxfreq_elmt]
    fft_freqs = fft_freqs[minfreq_elmt:maxfreq_elmt]

    peak_elmt = (argsort(abs(sig_fft)))[-1]
    
    pkfactor = np.max(np.abs(sig_fft))/np.sqrt(np.average(np.abs(sig_fft)**2))
    return [fft_freqs[peak_elmt], peak_elmt, pkfactor]

def remap_periodic(input_array, min_val, period = 2*pi):
    while len(input_array[input_array<min_val]) > 0:
        input_array[input_array<min_val] += period
    while len(input_array[input_array>=min_val+period]) > 0:
        input_array[input_array>=min_val+period] -= period
    return input_array

def list2bin(input_list):
    # we explicitly cast to int(), as numpy's integer type clashes with sqlalchemy
    return int(sum(2**array(input_list)))

def bin2list(input_value):
    output_list = []
    bin_index_str = _bin(input_value)[2:][::-1]
    for ind,i in enumerate(bin_index_str):
        if i == '1':
            output_list.append(ind)
    return output_list

def split_names(names, pad=' ',min_length=3):
    """ Given an array of strings, return an array of the part of the string
    (e.g. channel name) that varies, and optionally the prefix and suffix.
    The array of varying parts is first in the tuple in case others are not
    wanted.  This is used to make the x labels of phase plots simpler and smaller.
    e.g.

    >>> split_names(['MP01','MP10'],min_length=2)
    (['01', '10'], 'MP', '')

    The pad char is put on the end of shorter names - a better way would be
    to keep the end char the same, and pad in between the beginning and end
    the per channel part is at least min_length long.  
    This is not really needed, as the routine chooses the lenght so
    that the results are not ambiguous  (MP01,MP02 -> 1,2 but MP01,MP12 -> 01,12
    """
    # make a new array with elements padded to the same length with <pad>
    nms = []
    maxlen = max([len(nm) for nm in names])  # length of the longest name
    for nm in names:
        nmarr = [c for c in nm]
        while len(nmarr)< maxlen: nmarr.append(pad)
        nms.append(nmarr)
    
    # the following numpy array comparisons look simple, but require the name string
    # to be exploded into chars.  Although a single string can be interchangeably 
    # referred to as a string or array of chars, these arrays they have to be 
    # re-constituted before return.
    #
    #    for nm in nms:     # for each nm
    #find the first mismatch - first will be the first char of the extracted arr
    nms_arr=array(nms)
    first=0
    while (first < maxlen and
           (nms_arr[:,first] == nms_arr[0,first]).all()):
        first += 1

    # and the last        
    last = maxlen-1
    while ((last >= 0) and
           (nms_arr[:,last] == nms_arr[0,last]).all()):
        last -= 1


    # check for no mismatch        
    if first==maxlen: return(['' for nm in names], ''.join(nms[0]),'')
    # otherwise return, (no need for special code for the case of no match at all)
    if (1+last-first) < min_length:
        add_chars = min_length - (1+last-first)
        first = max(0, first-add_chars)
        print(first, last, add_chars, maxlen)
    return(([''.join(s) for s in nms_arr[:,first:last+1]],
            ''.join(nms_arr[0,0:first]),
            ''.join(nms_arr[0,last+1:maxlen+1])))

def make_title(formatstr, input_data, channum=None, at_dict = {}, min_length=3, raw_names=False):
    """ Return a string describing the shot number, channel name etc using
    a formatstr which refers to items in a dictionary (at_dict), assembled in
    this routine, based on input_data and an optional dictionary which
    contains anything not otherwise available in input_data

    """
##    at_dict.update({'shot': input_data.meta['shot']})
    exception = () if pyfusion.DBG() > 3 else Exception
    try:
        at_dict.update(input_data.meta)  # this gets all of it!


        if channum is None:
            name = ''
        else:
            if  isinstance(input_data.channels, list):
                chan = input_data.channels[channum]
            else:
                chan = input_data.channels
            if raw_names:
                name = chan.name
            else:
                name = chan.config_name
                
        at_dict.update({'units':chan.units})
        at_dict.update({'name': name})
# replace internal strings of non-numbers with a single .  a14_input03 -> 14.03
        short_name=''
        last_was_number = False
        discarded=''
        for c in name:  # start from the first char
            if c>='0' and c<='9': 
                short_name += c
                last_was_number=True
            else:  
                if last_was_number: short_name += '.'
                else: discarded += c
                last_was_number=False

                
        if len(short_name) <= min_length: 
            # if it fits, have the lot
            if len(name)<8: short_name=name
            # else allow 4 more chars - makes about 6-8 chars
            else: short_name = discarded[-4:] + short_name

        at_dict.update({'short_name': short_name})

        return(formatstr.format(**at_dict))
    except exception as ex:
        warn('in make_title for format="%s", at_dict=%s' % (formatstr, at_dict),
             exception=ex)
        return('')

if __name__ == '__main__':

# test program
    import doctest
    doctest.testmod()

    import pyfusion
    x=find_peaks([1,2,3,4,2,1,1,10,5,4,3,4,5])
    tb = np.linspace(0,1e-3,1000)
    signal = np.cos(2*np.pi*20e3*tb) + np.sin(2*np.pi*40e3*tb)
    (ip,fp,ap) = find_signal_spectral_peaks(tb, signal, debug=2)
    subdivide_interval([.5,1,1.2,2,3], debug=2)
