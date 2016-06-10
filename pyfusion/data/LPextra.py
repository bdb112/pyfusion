from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import pyfusion
from pyfusion.debug_ import debug_

def estimate_error(self, result, debug=1):
    if self.fitter.pcov is None:
        # Use the values in fitter.trace to backtrack the convergence
        meth = 'trace'
        trace = self.fitter.trace
        atrace = np.array(trace)  # array version - need both
        final_res = trace[-1][1]
        wgt = np.where(atrace[:, 1] > self.fitter.actual_fparams['track_ratio']*final_res)[0]
        if len(wgt) < 1:
            pyfusion.logging.warn('insufficient convergence to estimate errors')
            return(3 * [np.nan])
        ifrom = wgt[-1]
        vals = np.array([elt[0].tolist() for elt in trace[ifrom:]])
        est_errs = [(np.max(x)-np.min(x))/2. for x in vals.T]
    else:
        meth = 'cov'
        pcov = self.fitter.pcov
        s_sq = self.fitter.s_sq
        if s_sq>0 and pcov is not None:
            pcov = pcov * s_sq
        else:
            pcov = np.inf

        error = [] 
        for i in range(len(result[0])):
            try:
              error.append(np.absolute(pcov[i][i])**0.5)
            except:
              error.append( 0.00 )
        est_errs = np.array(error)

    errs = str('{meth} esterrs {e}'
               .format(e=', ' .join(['{v:.2g}'.format(v=v) for v in est_errs],),
                       meth = meth))
    if debug>1: print(errs)
    return(est_errs)


def lpfilter(t, i, v, lpf, debug=1, plot=0):
    """ Remove
    v is used only to determine the number of cycles
    """
    # watch out for odd numbers of samples
    if len(t)%2: 
        t=t[0:-1]
    ftv = np.fft.rfft(v)
    absftvlow = np.abs(ftv[1:20])  # exclude DC
    wothers = np.where(absftvlow > np.max(absftvlow)/100.)[0]
    if len(wothers) == 1:
        ncycles = wothers[0] + 1  # we excluded DC - so 0 is fundamental
    else:
        #  this problem can arise when using 'nice' FFT size - e.g. 2048
        msg = str('Not a clear number of cycles - assuming 1 [{harms}]'
                  .format(harms=', '.join(['{h}'.format(h=round(h/100,1)) 
                                           for h in absftvlow][0:4])))
        ncycles = 1
        pyfusion.logging.warn(msg)
        if debug>1: 
            print(msg)

    ft = np.fft.rfft(i)

    if lpf is None:
        f_filt = ft.copy()
    else:
        f_filt = ft*np.exp(-(np.arange(len(ft))/(1.7*ncycles*lpf))**2)

    half = chr(189).decode('latin1')
    if ncycles == 2:
        badharm = range(1, 40, 2)
        goodharm = range(ncycles, 5*ncycles, ncycles)
        frac = ['', half]
    else:
        badharm = []
        goodharm = range(1, 10)
        frac = ['']
    for n in badharm:
        f_filt[n] = 0
    i_filt = np.fft.irfft(f_filt)
    if np.shape(t) != np.shape(i_filt):
        raise ValueError('lpfilter - mismatched plot args')
    if plot > 2: plt.plot(t, i_filt)
    goodRMS = np.sqrt(np.sum(np.abs(ft)[goodharm]**2))
    contam = np.sqrt(np.sum(np.abs(ft)[badharm]**2))/goodRMS
    harms = unicode(',  '.join(['{n}{f}:{m:.0f}%'.decode('latin1')
                                .format(n=n//ncycles, f=frac[n % ncycles],
                                        m=100*np.abs(ft[n]/goodRMS))
                                for n in badharm
                                if np.abs(ft[n]/goodRMS) > 0.1]))

    extra = str('{h}th harm 3dB, {n} cycles, RMS contam={cpc:.0f}%'
                .format(h=lpf, cpc=contam*100, n=ncycles))
    if len(harms) > 10:
        extra += '\n'
    extra += ' [' + harms + ']'
    if debug>1:
        print(extra)
        print(len(i_filt), len(v))

    debug_(debug, 3, key='lpfilter')
    return(i_filt)

if __name__ == "__main__":
    ncycles = 2
    plt.figure()
    t = np.linspace(-.5, .49, 1000)  # .49 will generate 'not clear cycles'
    v = 50 * np.sin(2 * np.pi * ncycles * t)
    i = 0.02 * (1 - np.exp((v - 20)/50))
    ax = plt.gca()
    ax.plot(t, i, label='i raw')
    ax.plot(t, lpfilter(t, i, v, 4), label='i filt 4')
    ax2 = ax.twinx()
    ax2.plot(t, v, label='V')
    ax.legend(loc=2)
    ax2.legend()
    plt.show()
