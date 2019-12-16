from __future__ import print_function

import numpy as np
import pyfusion
from pyfusion.debug_ import debug_
from pyfusion.utils import wait_for_confirmation
from matplotlib import pyplot as plt

def find_shot_times(shot = None, diag = 'W7X_UTDU_LP10_I', threshold=0.2, margin=[.3,.7], debug=0, duty_factor=0.12, secs_rel_t1=False, exceptions=(LookupError), nsamples=2000):
    """ Return the actual interesting times in utc (absolute) for a given
    shot, based on the given diag.  
    secs_rel_t1: [False] - if True, return in seconds relative to trigger 1
    We use raw data to allow for both 1 and 10 ohm resistors (see above common mode sig)
    Tricky shots are [20171025,51] # no sweep), 1025,54 - no sweep or plasma
    See the test routine in __main__ when this is 'run' 
         
    Returns None if there is a typical expected exception (e.g. LookupError)
    Occasional problem using a probe signal if startup is delayed - try ECRH
    Would be good to make sure the start was before t1, but this code
    does not access t1 at the moment.

    debug=1 or more plots the result.
    """
    dev_name = "W7X"
    #  diag = W7X_UTDU_LP10_I  # has less pickup than other big channels
    dev = pyfusion.getDevice(dev_name)
    nsold = pyfusion.NSAMPLES
    pyfusion.RAW = 1  # allow for both 10 and 1 ohm sensing resistors
    # This should include a test for interactive use, so big save jobs
    # don't stall here
    if margin[0]<0:
        wait_for_confirmation('You will not save t1? with margin={m}\n {h}(n/q/y)'
                              .format(m=str(margin),h=__doc__))
    try:
        pyfusion.NSAMPLES = nsamples
        dev.acq.repair = -1
        save_db = pyfusion.DEBUG
        if plt.is_numlike(pyfusion.DEBUG):
            pyfusion.DEBUG -= 2  # lower debug level to allow us past here
        data = dev.acq.getdata(shot, diag, exceptions=())
    except exceptions as reason:  # return None if typical expected exception
        print('Exception suppressed: ', str(reason))
        return None
    except Exception as reason:
        print('Exception NOT suppressed: ', str(reason))
        raise
        
    finally:
        # this is executed always, even if the except code returns
        pyfusion.DEBUG = save_db
        pyfusion.NSAMPLES = nsold
        pyfusion.RAW = 0
        print('params restored')
        debug_(pyfusion.DEBUG, 3, key='find_shot_times')
                    
    if not isinstance(data.timebase[1], int) and hasattr(data, 'params') and 'diff_dimraw' in data.params:
        data.timebase = data.params['diff_dimraw'].cumsum()
    if len(np.unique(np.diff(data.timebase)))>1: # test for minmax reduction
        # requires a high DF than a fully sampled signal
        duty_factor = min(duty_factor * 4,0.9)
        
    tb = data.timebase
    tsamplen = tb[-1] - tb[0]
    for trial in range(40):
        wbig = np.where(np.abs(data.signal) > np.abs(threshold))[0]
        if len(wbig) < 5:
            threshold *= 0.8
            continue
        times = np.array([tb[wbig.min()], tb[wbig.max()]], dtype=np.int64)
        # fract_samples > 0.2 fract time avoids influence of spikes
        fract_time = (times[1] - times[0])/float(tsamplen)
        fract_samples = len(wbig)/float(len(tb))
        if debug>0: print('trial {t}, lentb {lentb}, thresh {thresh:.3f}, fract_time {fract_time:.3f}, fract_samples {fract_samples:.3f}, DF {DF:.3f}'
                          .format(t=trial, lentb=len(tb), thresh=threshold,
                                  fract_time=fract_time, fract_samples=fract_samples,
                                  DF =duty_factor))
        if fract_time > 0.95 and fract_samples/fract_time > duty_factor:
            threshold *= 1.2
            continue
        shortest = 0.2 * 1e9/tsamplen  # want to keep pulses of 0.2 sec even if on 20 sec stream
        if fract_time < min(0.05, shortest)  or fract_samples/fract_time < duty_factor:
            threshold *= 0.9
            continue
        break
    else:  # went through the whole loop (i.e. without success)
        pyfusion.utils.warn('Too few/many points above threshold on shot {shot}'.format(shot=str(shot)))
        if debug>1:
            plt.figure()
            data.plot_signals()
            plt.show()
        return None
    timesplus = np.array([times[0] - margin[0]*1e9, times[1] + margin[1]*1e9], dtype=np.int64)
    print('{sh}: shot length={dt}, {timesplus}'
          .format(sh=shot, dt=np.diff(times)/1e9, timesplus=(timesplus - tb[0])/1e9))
    if debug>0:
        plt.figure()  # need a new fig whilever we plt in absolute times
        data.plot_signals()
        plt.plot(timesplus, [threshold, threshold],'o--r')
        plt.plot(timesplus, [-threshold, -threshold],'o--r')
        plt.xlim(2*timesplus - times)  # double margin for plot
        plt.show()

    fact = 1
    if secs_rel_t1:
        utc_0 = data.params.get('utc_0', 0)
        if utc_0 == 0:
            raise LookupError("no t1 value, so can't select secs_rel_t1")
        else:
            fact = 1e-9
    else:
        utc_0 = 0
    print(fact, utc_0)
    return((timesplus - utc_0) * fact)

if __name__ == "__main__":
    print(find_shot_times([20170913, 27], debug=1))
    from pyfusion.data.shot_range import shot_range
    shot_list = [[20171024, 37]]  # vrey short
    shot_list.append([20171025,51]) # no sweep
    shot_list.append([20171025,54]) # no sweep no plasma
    shot_list += shot_range([20171025,1],[20171025,99])
    print('All the results')
    results = dict([[tuple(shot), find_shot_times(shot)] for shot in shot_list])
    import pprint
    known_duds = [ 1,  2,  3,  4, 28, 29, 31, 41, 43, 45, 46, 53, 54, 57]
    pprint.pprint(results)
    incorrect = [res[1] for res in results if results[res] is None and res[1] not in known_duds]
    print(' of these, the following {li} are incorrect'.format(li=len(incorrect)))
    pprint.pprint(incorrect)

    """  sort([res[1] for res in results if results[res] is None],axis=0)
    pyfusion.NSAMPLES=2000
    pyfusion.config.set('global','localdatapath','x')
    run -i pyfusion/examples/plot_shots shot_list"=[res for res in results if results[res] is None]" diag=W7X_UTDU_LP10_I
    """
