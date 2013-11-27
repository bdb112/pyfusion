""" signal processing peculiar to plasma devices - more general stuff in signal_processing.py
Simple criterion is if it imports pyfusion it should be here, if general, in signal_processing.
Important to separate from routines that initialise database, as they can't be
recompiled/reloaded easily during debugging.
""" 
import pyfusion
import pylab as pl
from numpy import average, max, cumsum, sqrt, mean, array
import numpy as np

def find_shot_times(dev, shot, activity_indicator=None, debug=0):
    """ Note: This is inside a try/except - errors will just skip over!! fixme
    From the channel specified in the expression "activity_indicator", determine
    the beginning and end of pulse.  A suitable expression is hard to
    find.  For example, density usually persists too long after the
    shot, and sxrays appear a little late in the shot.  The magnetics
    may be useful if magnet power supply noise could be removed.
    (had trouble with lhd 50628 until adj threshold ?start and end were at 0.5-0.6 secs )
    >>> import pyfusion
    >>> sh=pyfusion.core.get_shot(15043,activity_indicator="MP4")
    >>> print('start=%.3g, end=%.3g' % (sh.pulse_start, sh.pulse_end) )
    start=177, end=218
    >>> sh=pyfusion.core.get_shot(33372,activity_indicator="MP4")
    >>> print('start=%.3g, end=%.3g' % (sh.pulse_start, sh.pulse_end) )
    start=168, end=290
    """

    from pyfusion.data.signal_processing import smooth, smooth_n
    if debug>2: exception = None  # allow all exceptions to crash to debug
    else: exception = Exception

    if activity_indicator=="": 
        if pyfusion.OPT>5: 
            print(str(' No activity indicator connected to shot %d, ' 
                      'please consider implementing one to improve speed' % shot))
        return((pyfusion.settings.SHOT_T_MIN, pyfusion.settings.SHOT_T_MAX))

    diff_method = False
    try:  # if a single channel
        ch = dev.acq.getdata(shot, activity_indicator)

    except exception:
        if pyfusion.VERBOSE>0: print("using default activity indicator")

        if dev.name == 'HeliotronJ': 
            diff_method = True;
            cha = "MP3"
            chb = "MP1"

        elif dev.name == 'LHD': 
            diff_method = True;
            cha = "MP4"
            chb = "MP6"

    ## Assume the start baseline and the end baselines are different (e.g. MICRO01!)
    
    # for now, we hardwire in activity in MP1
    # later, change this to something like 'rms(pyf_hpn("MP1",2e3,4))>0.1'
    #  note: 15043 is a tricky test (bump at 290) (3v, 5us spike)
    threshold_type = True;
    level_type = False
    # the differential method should be useful for all,
    # but relies on the relative sensititivy of two channels to mains ripple
    # so only implement selectively.

    if not diff_method: 
        sig = ch[activity_indicator]
        timebase = ch.timebase
    else:
        activity_indicator = 'diff('+cha+ '-' +chb + ')'
        
    if level_type:
        n_avg = 10
        n_smooth = n_avg

        csum = cumsum(sig)
        # just the valid bit - signal_processing.smooth() does this better.
        sm_sig = (csum[2*n_smooth:] - csum[n_smooth:-n_smooth])/n_smooth

        maxpp = max(sm_sig)-min(sm_sig)
        threshold = max(0.005, maxpp/20)

    elif threshold_type: 
        n_avg = 300   # ripple is about 3ms (need to make this in phys units)
        n_smooth = n_avg

        if diff_method:
            # subtract two distant probes with similar power supply pickup.
            # distant increases phase diff hence real signal, and PS pickup will reduce if similar levels.
            ch1 = dev.acq.getdata(shot, cha)
            siga=ch1[cha]
            timebase = ch1.timebase
            ch2 = dev.acq.getdata(shot, chb)
            sigb=ch2[chb]
            tb2 = ch2.timebase
            if np.max(np.abs(tb2[0:10] - timebase[0:10]))> 1e-6:
                raise LookupError('timebases of {ca} and {cb} are different: '
                                  '\n {tb1}  \n{tb2}'
                                  .format(ca=cha, cb=chb, 
                                          tb1=timebase[0:10], tb2=tb2[0:10]))
            if pyfusion.VERBOSE>2: print("find_shot_times diff method, ids = %d, %d" % (id(siga), id(sigb)))
            sig = siga-sigb
            sm_sig=sqrt(smooth((sig-smooth(sig,n_smooth,keep=1))**2,n_smooth,keep=1))
            sm_sig[-n_smooth:]=sm_sig[-2*n_smooth:-n_smooth]
            tim=timebase
            threshold = 0.03   # good compromise is 0.02, 200 points, 1st order

        else:
            (inds, LP_sig) = smooth_n(sig,n_smooth,iter=4, indices=True,
                                      timebase=timebase)
            HP_sig = sig[inds] - LP_sig
            (tim,sm_sigsq) = smooth_n(HP_sig*HP_sig,n_smooth,
                                      timebase=timebase[inds])
            sm_sig = sqrt(sm_sigsq)
            threshold = 0.02   # good compromise is 0.02, 1500 points, 4th order

        maxpp = max(sm_sig)-min(sm_sig)

    start_bl = average(sm_sig[0:n_avg])
    end_bl = average(sm_sig[-n_avg:])

# if signal is quiet, but shows a contrast > 5, reduce threshold
    if maxpp < .1 and ((start_bl < maxpp/5) or (end_bl < maxpp/5)):
        threshold = maxpp/3

#    first_inds = (abs(sm_sig-start_bl) > threshold).nonzero()[0]
#    last_inds = (abs(sm_sig-end_bl) > threshold).nonzero()[0]
# New code is impulse proof - feature needs to last longer than one interval n_smooth
    first_inds=(smooth(abs(sm_sig-start_bl) > threshold, 2*n_smooth)>0.7).nonzero()[0]
    last_inds=(smooth(abs(sm_sig-end_bl) > threshold, 2*n_smooth)>0.7).nonzero()[0]
    
    if (debug>0) or pyfusion.VERBOSE>2: 
        fmt="%d: %s, threshold = %.3g, n_smooth=%d,"+\
            "n_avg=%d "
        fmt2="maxpp= %.3g, start_baseline=%.3g, end_baseline=%.3g,"+\
            " threshold=%.3g"
        info1=str(fmt % (shot, activity_indicator, threshold, n_smooth, n_avg))
        info2=str(fmt2 % (maxpp, start_bl, end_bl, threshold))
        print("activity indicator " + info1+'\n'+info2)

    if (debug>0) or (pyfusion.VERBOSE>2):  # plot before possible error signalled
        pl.plot(timebase, sig, 'c')
        pl.plot(tim, sm_sig,'b')
        pl.title('smoothed and raw signals used in finding active time of shot')
        pl.xlabel(info1+'\n'+info2)
        xr=pl.xlim()
        pl.plot([xr[0],mean(xr)], array([1,1])*start_bl)
        pl.plot([mean(xr),xr[1]], array([1,1])*end_bl)

    if len(first_inds) ==0 or  len(last_inds) ==0: 
        raise ValueError, str(
            'could not threshold the activity channel %s, %d start, %d end inds ' %
            (activity_indicator, len(first_inds), len(last_inds)))

        ## the first n_smooth is a correction for the lass of data in smoothing
        ## the last is a margin of error
        ## (have!) should replace this with actual corresponding time
        #start_time=ch.timebase[max(0,min(first_inds)+n_smooth-n_smooth)]
        #end_time=ch.timebase[min(len(sig)-1,max(last_inds)+
        #                                n_smooth+n_smooth)]
                                     
    start_time = tim[min(first_inds)]
    end_time = tim[max(last_inds)]
    end_time = min(end_time,timebase[-1])  

    if pyfusion.VERBOSE>2: print(end_time, last_inds)

    if (debug>0) or pyfusion.VERBOSE>4: # two crosses mark the endpoints
        pl.plot([start_time,end_time],[start_bl, end_bl], " +k", markersize=20, mew=0.5)
        pl.plot([start_time,end_time],[start_bl, end_bl], " ok", mfc='None', markersize=20, mew=1.5)
        # scatter is "out of date"  - integer width, different conventions and is hidden 
        # underneath plots
        # pl.scatter([start_time,end_time],[start_bl, end_bl], s=100, marker="+", linewidth=2)

    if pyfusion.VERBOSE>0: 
        print("found start time on %d of %.5g, end = %.5g using %s" %
              (shot, start_time, end_time, activity_indicator))

    return(start_time, end_time)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
