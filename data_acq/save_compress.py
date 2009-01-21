from numpy import max, std, array, min, sort, diff, unique, size, mean, mod,\
    log10, int16, int8, uint16, uint8
try: 
    from pyfusion.hacked_numpy_io import savez
except:
    print("Couldn't load pyfusion.hacked_numpy_io.py, compression much less effective")
try:
    from pyfusion.pyfusion_settings import VERBOSE as verbose
except: 
    verbose=1
    print("assuming verbose output")

    from numpy import savez

def discretise_array(arr, eps=0, bits=0, maxcount=0, verbose=0):
    """
    Return an integer array and scales etc in a dictionary 
    - the dictionary form allows for added functionaility.
    If bits=0, find the natural accuracy.  eps defaults to 3e-6, and 
    is the error relative to the larest element, as is maxerror.
    """
    if eps==0: eps=3e-6
    if maxcount==0: maxcount=10
    count=1
    ans=try_discretise_array(arr, eps=eps,bits=bits, verbose=verbose)
    initial_deltar=ans['deltar']
    # look for timebase, because they have the largest ratio of value to
    #  step size, and are the hardest to discretise in presence of repn err.
    # better check positive!  Could add code to handle negative later.
    if initial_deltar>0:
        # find the largest power of 10 smaller than initial_deltar
        p10r=log10(initial_deltar)
        p10int=int(100+p10r)-100   # always round down
        ratiop10=initial_deltar/10**p10int
        eps10=abs(round(ratiop10)-ratiop10)
        if verbose>3: print("ratiop10, p10r, eps10, p10int, initial_deltar", 
              ratiop10, p10r, eps10, p10int, initial_deltar)
        if eps10<3e-3*ratiop10: 
            initial_deltar=round(ratiop10)*10**p10int
            if verbose>2: print("trying an integer x power of ten")
            ans=try_discretise_array(arr, eps=eps,bits=bits, 
                                     deltar=initial_deltar, verbose=verbose)
            initial_deltar=ans['deltar']

    while (ans['maxerror']>eps) and (count<maxcount):
        count+=1
        # have faith in our guess, assume problem is that step is
        # not the minimum.  e.g. arr=[1,3,5,8] 
        #          - min step is 2, natural step is 1
        ans=try_discretise_array(arr, eps=eps,bits=bits,deltar=initial_deltar/count, verbose=verbose)
        
    if verbose>0: print("integers from %d to %d, delta=%.5g" % (\
            min(ans['iarr']), max(ans['iarr']), ans['deltar']) )    
    return(ans)
#    return(ans.update({'count':count})) # need to add in count

def try_discretise_array(arr, eps=0, bits=0, deltar=None, verbose=0):
    """
    Return an integer array and scales etc in a dictionary 
    - the dictionary form allows for added functionaility.
    If bits=0, find the natural accuracy.  eps defaults to 1e-6
    """
    if verbose>2: import pylab as pl
    if eps==0: eps=1e-6
    mono= (diff(arr)>0).all()  # maybe handy later? esp. debugging
    if deltar==None: 
        data_sort=unique(arr)
        diff_sort=sort(diff(data_sort))  # don't want uniques because of noise

    # with real representation, there will be many diffs ~ eps - 1e-8 
    # or 1e-15*max - try to skip over these
    #  will have at least three cases 
    #    - timebase with basically one diff and all diffdiffs in the noise    
    #    - data with lots of diffs and lots of diffdiffs at a much lower level
        min_real_diff_ind=(diff_sort > max(diff_sort)/1e4).nonzero()
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

        # When the step size is within a few orders of represeantaion
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

        deltar=mean(diff_sort[0:last_small_diff_diffs_ind])
        peaknoise = max(abs(diff_sort[0:last_small_diff_diffs_ind] -
                                deltar))
        rmsnoise = std(diff_sort[0:last_small_diff_diffs_ind] -
                                deltar)
        pktopk=max(arr)-min(arr)
        if (verbose>0) or (peaknoise/pktopk>1e-7): 
            print('over avering interval relative numerical noise ~ %.2g pk, %.2g RMS' % 
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

    if min(iarr>=0):
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
            
        elif max(iarr)<8192: 
            iarr=iarr.astype(int16)
            if verbose>1: print('using 16 bit ints')
                
    return({'iarr':iarr, 'maxerror':maxerr, 'deltar':deltar, 'minarr':min(arr),
            'intmax':max(iarr)})

def discretise_signal(timebase=None, signal=None, parent_element=None,
                      eps=0, verbose=0, 
                      delta_encode_time=True, 
                      delta_encode_signal=False,
                      filename=None):
    """  a function to return a dictionary from signal and timebase, with 
    relative accuracy eps, optionally saving if filename is defined. 
    Achieves a factor of >10x on MP1 signal 33373 using delta_encode_time=True
    Delta encode on signal is not effective for MP and MICROFAST (.005% worse)
    Probably should eventually separate the file write from making the 
    dictionary.  Intended to be aliased with loadz, args slightly different.
    There is no dependence on pyfusion.
    """
    from numpy import remainder, mod, min, max, \
        diff, mean, unique, append

    dat=discretise_array(signal,eps=eps,verbose=verbose)
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
        rawsignal=append(rawsignal[0],diff(rawsignal))
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
    if filename!=None:
        if verbose>0: print('Saving as %s' % filename)
        savez(filename, timebaseexpr=timebaseexpr, signalexpr=signalexpr,
              parent_element=parent_element,
              rawsignal=dat['iarr'], rawtimebase=rawtimebase, version=100)    


def newload(filename, verbose=verbose):
    """ Intended to replace load() in numpy
    """
    from numpy import load as loadz
    from numpy import cumsum
    dic=loadz(filename)
#    if dic['version'] != None:
#    if len((dic.files=='version').nonzero())>0:
    if len(dic.files)>3:
        if verbose>0: print ("version %d" % (dic['version']))
    else: 
        if verbose>0: print("version 0: simple")
        return(dic)  # quick, minimal return

    if verbose>0: print('file contains %s' % dic.files)
    signalexpr=dic['signalexpr']
    timebaseexpr=dic['timebaseexpr']
# savez saves ARRAYS always, so have to turn array back into scalar    
    exec(signalexpr.tolist())
    exec(timebaseexpr.tolist())
    return({"signal":signal, "timebase":timebase, "parent_element": dic['parent_element']})



def test_compress(file=None, verbose=0, eps=0, debug=False, maxcount=0):
    """ Used in developing the save compress routines.  Not tested since then
    >>>>test_compress()
    """
    print("Testing %s" % file)
    if file==None: file='18993_densitymediaIR.npz'
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