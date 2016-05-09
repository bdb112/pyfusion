""" 
ported from the old pyfusion:

A "smart compression" replacement for savez, assuming data is quantised.
The quantum is found, and the data replaced by a product of and integer sequence
and quantum with offset.  delta encoding is optional and often saves space.
The efficiency is better than bz2 of ascii data for individual channels, and
a little worse if many channels are lumped together with a common timebase
in the bz2 ascii format, because save_compress stores individual timebases.
$ wc -c /f/python/local_data/027/27999_P*  2300176 total

At the moment (2010), save_compress is not explicitly implemented - it is 
effected by calling discretise_signal() with a filename argument.

July 2009 - long-standing error in delta_encode_signal fixed (had not been 
usable before)

"""
from numpy import max, std, array, min, sort, diff, unique, size, mean, mod,\
    log10, int16, int8, uint16, uint8
import numpy as np

import pyfusion
from pyfusion.debug_ import debug_
  
try: 
    # now in numpy
    # from pyfusion.hacked_numpy_io import savez
    from numpy import savez_compressed as savez
except:
    print("couldn't load savez_compressed ")

# this connection with pyfusion.settings ONLY captures the value at startup - 
# i.e. doesn't respond to changes from within python
try:
    from pyfusion.pyfusion_settings import VERBOSE as verbose
except: 
    verbose=1
    print("assuming verbose output")

    from numpy import savez_compressed

def discretise_array(arrin, eps=0, bits=0, maxcount=0, verbose=None, delta_encode=False):
    """
    Return an integer array and scales etc in a dictionary 
    - the dictionary form allows for added functionaility.
    If bits=0, find the natural accuracy.  eps defaults to 3e-6, and 
    is the error relative to the larest element, as is maxerror.
    """
    verbose = pyfusion.VERBOSE if verbose is None else verbose
    if eps==0: eps=3e-6
    if maxcount==0: maxcount=10
    count=1
    wnan = np.where(np.isnan(arrin))[0]
    notwnan = np.where(np.isnan(arrin) == False)[0]
    if len(wnan) == 0: 
        arr = arrin
    else:
        print('{n} nans out of {l}'.format(n=len(wnan), l=len(arrin)))
        arr = arrin[notwnan]

    ans=try_discretise_array(arr, eps=eps,bits=bits, verbose=verbose, 
                             delta_encode=delta_encode)
    initial_deltar=ans['deltar']
    # try to identify the timebase, because they have the largest ratio of value to
    #  step size, and are the hardest to discretise in presence of repn err.
    # better check positive!  Could add code to handle negative later.
    if initial_deltar>0:
        # find the largest power of 10 smaller than initial_deltar
        p10r=log10(initial_deltar)
        p10int=int(100+p10r)-100   # always round down
        ratiop10=initial_deltar/10**p10int
        eps10=abs(round(ratiop10)-ratiop10)
        if verbose>3: print("ratiop10=%g, p10r=%g, eps10=%g, p10int=%d, initial_deltar=%g" % 
              (ratiop10, p10r, eps10, p10int, initial_deltar))
        if eps10<3e-3*ratiop10: 
            initial_deltar=round(ratiop10)*10**p10int
            if verbose>2: print("timebase: trying an integer x power of ten")
            ans=try_discretise_array(arr, eps=eps,bits=bits, 
                                     deltar=initial_deltar, verbose=verbose, 
                                     delta_encode=delta_encode)
            initial_deltar=ans['deltar']

    while (ans['maxerror']>eps) and (count<maxcount):
        count+=1
        # have faith in our guess, assume problem is that step is
        # not the minimum.  e.g. arr=[1,3,5,8] 
        #          - min step is 2, natural step is 1
        ans=try_discretise_array(arr, eps=eps,bits=bits,
                                 deltar=initial_deltar/count, verbose=verbose,
                                 delta_encode=delta_encode)
        
    if verbose>0: print("integers from %d to %d, delta=%.5g" % (\
            min(ans['iarr']), max(ans['iarr']), ans['deltar']) )    
    if len(wnan)>0:
        dtyp = ans['iarr'].dtype
        maxint = np.iinfo(dtyp).max
        if maxint==ans['iarr'].any():
            print('******** warning: save_compress already using maxint - now defining it as a nan!')

        orig_iarr =  ans['iarr']  # for debugging
        full_iarr = np.zeros(len(arrin), dtype=dtyp)
        full_iarr[notwnan] = ans['iarr']
        full_iarr[wnan] = maxint
        ans['iarr'] = full_iarr
    debug_(pyfusion.DEBUG, 3, key='discretise')
    return(ans)
#    return(ans.update({'count':count})) # need to add in count

def try_discretise_array(arr, eps=0, bits=0, deltar=None, verbose=0, delta_encode=False):
    """
    Return an integer array and scales etc in a dictionary 
    - the dictionary form allows for added functionality.
    If bits=0, find the natural accuracy.  eps defaults to 1e-6
    """
    if verbose>2: import pylab as pl
    if eps==0: eps=1e-6
    mono= (diff(arr)>0).all()  # maybe handy later? esp. debugging
    if deltar is None: 
        data_sort=unique(arr)
        diff_sort=sort(diff(data_sort))  # don't want uniques because of noise
        if size(diff_sort) == 0: diff_sort = [0]  # in case all the same
    # with real representation, there will be many diffs ~ eps - 1e-8 
    # or 1e-15*max - try to skip over these
    #  will have at least three cases 
    #    - timebase with basically one diff and all diffdiffs in the noise    
    #    - data with lots of diffs and lots of diffdiffs at a much lower level

        min_real_diff_ind=(diff_sort > max(diff_sort)/1e4).nonzero()
        if size(min_real_diff_ind) == 0: min_real_diff_ind = [[0]]
#      min_real_diff_ind[0] is the array of inidices satisfying that condition 
        if verbose>1: print("min. real difference indices = ", min_real_diff_ind)
        #discard all preceding this
        diff_sort=diff_sort[min_real_diff_ind[0][0]:]
        deltar=diff_sort[0]
        diff_diff_sort=diff(diff_sort)
        # now look for the point where the diff of differences first exceeds half the current estimate of difference
        
        # the diff of differences should just be the discretization noise 
        # by looking further down the sorted diff array and averaging over
        # elements which are close in value to the min real difference, we can
        # reduce the effect of discretization error.
        large_diff_diffs_ind=(abs(diff_diff_sort) > deltar/2).nonzero()
        if size(large_diff_diffs_ind) ==0:
            last_small_diff_diffs_ind = len(diff_sort)-1
        else: 
            first_large_diff_diffs_ind = large_diff_diffs_ind[0][0]
            last_small_diff_diffs_ind = first_large_diff_diffs_ind-1
            
        # When the step size is within a few orders of representation
        # accuracy, problems appear if there a systematic component in
        # the representational noise.

        # Could try to limit the number of samples averaged over,
        # which would be very effective when the timebase starts from
        # zero.  MUST NOT sort the difference first in this case!
        # Better IF we can reliably detect single rate timebase, then
        # take (end-start)/(N-1) if last_small_diff_diffs_ind>10:
        # last_small_diff_diffs_ind=2 This limit would only work if
        # time started at zero.  A smarter way would be to find times
        # near zero, and get the difference there - this would work
        # with variable sampling rates provided the different rates
        # were integer multiples.  another trick is to try a power of
        # 10 times an integer. (which is now implemented in the calling routine)

        # Apr 2010 - fixed bug for len(diff_sort) ==  1  +1 in four places
        # like [0:last_small_diff_diffs_ind+1] - actually a bug for all, only
        # obvious for len(diff_sort) ==  1
        if pyfusion.DBG(): 
            print('last_small_diff_diffs_ind', last_small_diff_diffs_ind)
        debug_(pyfusion.DEBUG, 2, key='last_small')
        if last_small_diff_diffs_ind < 0:
            print('last_small_diff_diffs_ind = {lsdd} - error?  continuing...'
                  .format(lsdd=last_small_diff_diffs_ind))
            deltar, peaknoise, rmsnoise = 0, 0, 0
        else:
            deltar=mean(diff_sort[0:last_small_diff_diffs_ind+1])
            peaknoise = max(abs(diff_sort[0:last_small_diff_diffs_ind+1] -
                                deltar))
            rmsnoise = std(diff_sort[0:last_small_diff_diffs_ind+1] -
                                deltar)
        pktopk=max(arr)-min(arr)
        if (verbose>0) or (peaknoise/pktopk>1e-7): 
            print('over averaging interval relative numerical noise ~ %.2g pk, %.2g RMS' % 
                  (peaknoise/pktopk, rmsnoise/pktopk))

        if verbose>2: 
            st=str("averaging over %d diff diffs meeting criterion < %g " % 
                  (last_small_diff_diffs_ind, deltar/2 ))
            print(st)
            pl.plot(diff_sort,hold=0)
            pl.title(st)
            pl.show()
        if verbose>10: 
            dbg=0
            dbg1=1/dbg  # a debug point

    if verbose>1: print('seems like minimum difference is %g' % deltar)
    iarr=(0.5+(arr-min(arr))/deltar).astype('i')
    remain=iarr-((arr-min(arr))/deltar)
    remainck=mod((arr-min(arr))/deltar, 1)

# remain is relative to unit step, need to scale back down, over whole array
    maxerr=max(abs(remain))*deltar/max(arr)
# not clear what the max expected error is - small for 12 bits, gets larger quicly
    if (verbose>2) and maxerr<eps: print("appears to be successful")
    if verbose>0: print('maximum error with eps = %g, is %g, %.3g x eps' % (eps,maxerr,maxerr/eps))

   # only use unsigned ints if we are NOT delta_encoding and signal >0
    if (delta_encode == False and min(iarr)>=0):
        if max(iarr)<256: 
            iarr=iarr.astype(uint8)
            if verbose>1: print('using 8 bit uints')
            
        elif max(iarr)<16384: 
            iarr=iarr.astype(uint16)
            if verbose>1: print('using 16 bit uints')
                
    else:
        if max(iarr)<128: 
            iarr=iarr.astype(int8)
            if verbose>1: print('using 8 bit ints')
            
        elif max(iarr)<8192:   # why is this so conservative?  I would think 32766
            iarr=iarr.astype(int16)
            if verbose>1: print('using 16 bit ints')
    # if not any of the above, stays as an int32
                
    return({'iarr':iarr, 'maxerror':maxerr, 'deltar':deltar, 'minarr':min(arr),
            'intmax':max(iarr)})

def discretise_signal(timebase=None, signal=None, parent_element=array(0),
                      eps=0, verbose=0, params={},
                      delta_encode_time=True, 
                      delta_encode_signal=False,
                      filename=None):
    """a function to return a dictionary from signal and timebase, with 
    relative accuracy eps, optionally saving if filename is defined. 
    Achieves a factor of >10x on MP1 signal 33373 using delta_encode_time=True
    Delta encode on signal is not effective for MP and MICROFAST (.005% worse)
    Probably should eventually separate the file write from making the 
    dictionary.  Intended to be aliased with loadz, args slightly different.
    There is no dependence on pyfusion.  Version 101 adds time_unit_in_seconds
    version 102 adds utc, raw, 103 after correction of probes 11-20
    Note: changed to parent_element=array(0) by default - not sure what this is!
    """
    from numpy import remainder, mod, min, max, \
        diff, mean, unique, append
    from pyfusion.debug_ import debug_

    debug_(pyfusion.DEBUG, 2, key='discretise_signal')

    dat=discretise_array(signal,eps=eps,verbose=verbose, delta_encode=delta_encode_signal)
#    signalexpr=str("signal=%g+rawsignal*%g" % (dat['minarr'], dat['deltar']))
# this version works here and now - need a different version to work in loadz
#    signalexpr=str("signal=%g+%s*%g" % (dat['minarr'], "dat['iarr']",
#                                        dat['deltar']))

# this version is designed to be evaluated on dictionary called dic
    rawsignal=dat['iarr']
    signalexpr=str("signal=%g+%s*%g" % (dat['minarr'], "dic['rawsignal']",
                                        dat['deltar']))
    if delta_encode_signal: 
#        need to maintain the first element and the length
        tempraw=append(rawsignal[0],diff(rawsignal))
        rawsignal = tempraw
        signalexpr=str("signal=%g+%s*%g" % (dat['minarr'], 
                                                "cumsum(dic['rawsignal'])",
                                                dat['deltar']))
# saving bit probably should be separated
    tim=discretise_array(timebase,eps=eps,verbose=verbose)
    rawtimebase=tim['iarr']
    timebaseexpr=str("timebase=%g+%s*%g" % (tim['minarr'], "tim['iarr']",
                                            tim['deltar']))
# this version 
    timebaseexpr=str("timebase=%g+%s*%g" % (tim['minarr'], "dic['rawtimebase']",
                                          tim['deltar']))
    if delta_encode_time: 
#        need to maintain the first element and the length
        rawtimebase=append(rawtimebase[0],diff(rawtimebase))
        timebaseexpr=str("timebase=%g+%s*%g" % (tim['minarr'], 
                                                "cumsum(dic['rawtimebase'])",
                                                tim['deltar']))
    # This only works for NON delta encoded.
    wnan = np.where(np.isnan(timebase))[0]
    if len(wnan)>0:
        timebaseexpr = timebaseexpr + ";maxint = np.iinfo(dic['rawtimebase'].dtype).max;wnan = np.where(maxint == dic['rawtimebase'])[0];if len(wnan)>0:;    timebase[wnan]=np.nan".replace(';','\n')

    if verbose>4: 
        import pylab as pl
        pl.plot(rawsignal)

    if filename!=None:
        if verbose>0: print('========> Saving as %s <=======' % filename)
        # time_unit_in_seconds is automatically set by a fudge - won't work for W7-X Op2
        tus=[.001,1][max(abs(array(timebase)))<100]

        savez_compressed(filename, timebaseexpr=timebaseexpr, 
                         signalexpr=signalexpr, params=params,
                         parent_element=parent_element, time_unit_in_seconds=tus,
                         rawsignal=rawsignal, rawtimebase=rawtimebase, version=103)    


def newload(filename, verbose=verbose):
    """ Intended to replace load() in numpy
    This is being used with nan data.  The version in data/base.py is closer to
    python 3 compatible, but can't deal with the nans yet.
    """
    from numpy import load as loadz
    from numpy import cumsum
    dic=loadz(filename)
#    if dic['version'] != None:
#    if len((dic.files=='version').nonzero())>0:
    if len(dic.files)>3:
        if verbose>2: print ("local v%d " % (dic['version'])),
    else: 
        if verbose>2: print("local v0: simple "),
        return(dic)  # quick, minimal return

    if verbose>2: print(' contains %s' % dic.files)
    # savez saves ARRAYS always, so have to turn array back into scalar    
    signalexpr=dic['signalexpr'].tolist()
    timebaseexpr=dic['timebaseexpr'].tolist()
    # fixup for files written with np.nan removal and and cumsum
    if ('cumsum' in timebaseexpr) and ('np.nan' in timebaseexpr):
        print('!!!!!!!!!!!! fixup of nans with cumsum !!!!!!!!!!!!!!!!!!')
        timebaseexpr = timebaseexpr.replace("timebase=",
                             "temp=").replace("*2e-06","\ntimebase=temp*2e-06")
        timebaseexpr = timebaseexpr.replace("== dic['rawtimebase']","== temp")

    exec(signalexpr)
    exec(timebaseexpr)
    retdic = {"signal":signal, "timebase":timebase, "parent_element":
              dic['parent_element']}

    if 'params' in dic: retdic.update({"params": dic['params'].tolist()})
    return(retdic)

    # return({"signal":signal, "timebase":timebase, "parent_element": dic['parent_element']})

def save_compress(timebase=None, signal=None, filename=None, *args, **kwargs):
    """ save a signal and timebase into a compress .npz file.  See arglist
    of discretise_signal.
    Example:
    >>> sig=[1,2,1,2] ; tb=[1,2,3,4] # need this only for later comparison
    >>> save_compress(timebase=tb, signal=sig, filename='junk')
    >>> readback=newload('junk.npz',verbose=0)
    >>> if (readback['signal'] != sig).any(): print 'error in save/restore'
    
"""
    discretise_signal(timebase=timebase, signal=signal, filename=filename, *args, **kwargs)
## tack the discretise_signal doc on the end
    save_compress.__doc__ += "\n  Calls discretise_signal,\n   which is " + discretise_signal.__doc__

def test_compress(file=None, verbose=0, eps=0, debug=False, maxcount=0):
    """ Used in developing the save compress routines.  Not tested since then
    >>> test_compress()
    Looks like it only saves the time series, not the rest.
    """
    from numpy import load as loadz
    print("Testing %s" % file)
    if file is None: file='18993_densitymediaIR.npz'
    test=loadz(file)
    stat=os.stat(file)

    if verbose > 0: print("=========== testing signal compression ===========")
    sig=discretise_array(test['signal'],eps=eps,verbose=verbose,maxcount=maxcount)
    if verbose > 0: print("=========== testing timebase compression===========")
    tim=discretise_array(test['timebase'],eps=eps,verbose=verbose)
    print('  File length %d bytes, %d samples, %.3g bytes/sample' % (
            stat.st_size ,len(sig['iarr']),
            float(stat.st_size)/len(sig['iarr'])))
    temp='temp.npz'
    savez(temp,sig['iarr'])
    print("    compressed to %d bytes" % os.stat(temp).st_size)
    savez(temp,diff(sig['iarr']))
    print("      differences compressed to %d bytes" % os.stat(temp).st_size)
    savez(temp,diff(diff(sig['iarr'])))
    print("      double differences compressed to %d bytes" % os.stat(temp).st_size)
    print("    time compressed to %d bytes" % os.stat(temp).st_size)
    savez(temp,diff(tim['iarr']))
    print("      difference compressed to %d" % os.stat(temp).st_size)
    savez(temp,diff(diff(tim['iarr'])))
    print("      double difference compressed to %d" % os.stat(temp).st_size)
    if debug: xx=1/0

if __name__ == "__main__":
    import doctest
    doctest.testmod()
