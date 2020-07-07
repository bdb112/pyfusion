""" Complete processing of swept Langmuir data, using two classes.
New version allows choice of algorithm and adds more params
This supersedes the example in examples/process_swept_Langmuir.py
 
pyfusion v0.7.0 - estimate error, include a lpfilter, resid normed to I0
This version assumes that all required sweep data are in a multi-channel diagnostic (dataset)
 - the other version (should not be in the repo) grabs sweepV data as required

See main README.rst for more recent changes 
 - see also the docs of the main method :py:meth:`~Langmuir_data.process_swept_Langmuir`
 - The convenience script examples/run_process_Langmuir (run_process_TDLP.py etc)
   runs a list of shots for both limiter segments with suitable inputs

See also get_LP_data for tweaking fit algorithms - grabs v/i data from plots to play with

    Example of step by step operation and tuning:

    >>> run pyfusion/data/process_swept_Langmuir
    >>> LP.process_swept_Langmuir()
    >>> LP.write_DA('20160310_9_L57')
    >>> # then to tune mask:
    >>> from pyfusion.data.DA_datamining import Masked_DA, DA
    >>> myDA=DA('20160310_9_L57',load=1)  # just to be sure it is a good DA
    >>> myDA.masked=Masked_DA(['Te','I0','Vf'], baseDA=myDA)
    >>> myDA.da['mask']=(myDA['resid']/abs(myDA['I0'])<.35) & (myDA['nits']<100)
    >>> clf();plot(myDA.masked['Te']);ylim(0,100)


"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin
import os
import pyfusion
from pyfusion.utils.time_utils import utc_ns
from pyfusion.utils import wait_for_confirmation
import pyfusion.conf as conf
from scipy.fftpack import hilbert
from pyfusion.data import amoeba
from scipy.optimize import leastsq
import time
# from pyfusion.data.LPextra  # imported as needed
from pyfusion.debug_ import debug_
from pyfusion.utils.primefactors import nice_FFT_size
from pyfusion.utils import boxcar, rotate
from pyfusion.conf.utils import get_config_as_dict

debug = 0


def AC(x):
    ns = len(x)
    return(x - np.average(x * np.blackman(ns))/np.average(np.blackman(ns)))


def fixup(da=None, locs=None, newmask=None, suppress_ne=None, newpath='/tmp'):
    """ some miscellaneous post processing mainly to set the time offset of ECH
need to run plot_LP2d.py first.
sets t_zero, and tweaks the Nan coverage

Args: 
  da: a dictionary of arrays (DA) file.
  locs: a dictionary containing the variable dtprobe, usually locals()
  suppress_ne - comma delimited list of channels whose ne we don't believe e.g. area changed
    """
    if locs is None:
        raise ValueError('Need two args, first is an open da, second is locals()) containing variable dtprobe')
    da['info']['t_zero'] = -locs['dtprobe']
    da['info']['t_zero_utc_ns'] = (da['info']['params']['i_diag_utc'][0] +
                                   int(da['info']['t_zero']*1e9))
    if newmask is not None:
        if np.shape(newmask) != np.shape(da['mask']):
            raise ValueError('newmask shape {sn} is different to existing {se}'
                             .format(sn=np.shape(newmask), 
                                     se=np.shape(da['mask'])))

    if suppress_ne is not None:
        for (c, ch) in enumerate(da.infodict['channels']):
            if np.any([sup in ch for sup in suppress_ne]):
                da['mask'][:, c] = 0

    newname = os.path.join(newpath, os.path.split(da.name)[-1])
    print('saving as ' + newname)
    da.save(newname)
"""
# In order to make leastsq work we THOUGHT we needed these OUT of a class
# No so! see below
def LPchar(v, Te, Vf, I0):
    # hard to add a series resistance here as fun returns I as a fn of V
    return(I0 * (1 - exp((v-Vf)/Te)))

def residuals(params, y, x):
    Te, Vf, I0 = params
    err = LPchar(x, Te, Vf, I0) - y
    # return np.sqrt(np.abs(err))
    return err
"""

class LPfitter():
    """
    Mainly used by :py:meth:`~Langmuir_data.process_swept_Langmuir` but also stands alone.
    Assumes the i and v are in arrays already.  
    See `:py:meth:LPfitter.__init__` for more info
    """
    def __init__(self, i, v,  fit_params=None, plot=1, verbose=1, parent=None, debug=1):
        """
This is the code presently used to default fit params

    >>> self.fit_params = dict(maxits=300, alg='leastsq', ftol=1e-4, xtol=1e-4, Lnorm=1.2, track_ratio=1.2, lpf=None)

Safer to type this to see the actual defaults if the code has changed:
    >>> from pyfusion.data.process_swept_Langmuir import LPfitter
    >>> LPfitter(None, None).fit_params

        """
        self.i = i
        self.v = v
        self.debug = debug
        self.verbose = verbose
        self.plot = plot
        # this will be the default for fit_params 
        #   overridden by arg to __init__ and then 
        #   that is overridden by arg to fit
        self.fit_params = dict(maxits=300, alg='leastsq', ftol=1e-4, xtol=1e-4, Lnorm=1.2, track_ratio=1.2, lpf=None)
        if fit_params is not None: # override with init args
            self.fit_params.update(fit_params)
        self.parent = parent
        self.trace = []
        if (self.fit_params['alg'] == 'leastsq' 
            and self.fit_params['Lnorm']<1.2):
            pyfusion.logging.warn('leastsq behaves strangely near Lnorm=1')

    def LPchar(self, v, Te, Vf, I0):
        # hard to add a series resistance here as fun returns I as a fn of V
        return(I0 * (1 - exp((v-Vf)/Te)))

    def plotchar(self, v, Te, Vf, I0, alpha=1, col='g', linewidth=2):
        """ plot a Langmuir characteristic """
        varr = np.linspace(np.min(v), np.max(v))
        plt.plot(varr, self.LPchar(varr, Te, Vf, I0), 'k', linewidth=linewidth+1)
        plt.plot(varr, self.LPchar(varr, Te, Vf, I0), col, linewidth=linewidth)

    def error_fun(self, vars, data):
        """ return RMS error """
        Te, Vf, I0 = vars
        v = data['v']
        i = data['i']
        Lnorm = self.actual_fparams['Lnorm']
        varr = np.linspace(np.min(v), np.max(v))
        if self.plot > 3:
            # note - alpha chosen according to number of lines (maxits)
            #  We use the maxits value here via actual_fparams
            # we would prefer to use the actual number used
            plt.plot(varr, self.LPchar(varr, Te, Vf, I0), 'g', linewidth=2,
                     alpha=min(1, 1./np.sqrt(self.actual_fparams['maxits'])))
        #  err = np.sqrt(np.mean((i - LPchar(v, Te, Vf, I0))**2))
        err = np.power(np.mean(np.abs(i - self.LPchar(v, Te, Vf, I0))**Lnorm), 1./Lnorm)
        if self.debug > 4:
            print('error fun:'+', '.join(['{k} = {v:.3g}'.format(k=var, v=val)
                             for (var, val) in dict(Te=Te, Vf=Vf, I0=I0, resid=err).iteritems()]))
        self.trace.append([np.copy(vars), err])
        return(-err)  # amoeba is a maximiser, hence - sign


    def residuals(self, params, x, y):
        Te, Vf, I0 = params
        err = self.LPchar(x, Te, Vf, I0) - y
        Lnorm = self.fit_params['Lnorm']
        self.trace.append([params.copy(), np.power(np.mean(np.abs(err)**Lnorm), 1./Lnorm)])
        if  Lnorm != 2:  # leastsq routine will square the residual
            err = np.power(np.abs(err), Lnorm/2.)
        return err
        
    def fit(self, init=None, plot=None, fit_params=None, default_init=dict(Te=50, Vf=15, I0=None)):
        """ Perform a curve fit operation with the given parameter and initial
        value.  'None' invokes the fit_params determined at creation
        of the class, and for 'init', the default_init

        The final parameters actually used are saved in actual_params

        """
        # Takes about 2ms/it for 2500 points, and halving this only saves 10%
        # if maxits is None:  # this trickiness makes it work, but not pythonic.
        #    maxits = globals()['maxits']
        if init is None:
            init = default_init
        else: 
            default_init.update(init)
            init = default_init

        self.trace = []
        if plot is None:
            plot = self.plot

        # if None, get the defaults from the __init__
        if fit_params is None:
            if self.fit_params is not None:
                fit_params = self.fit_params
            else:
                raise ValueError('Need fit params in LPfitter class or call to .fit')
        self.actual_fparams = self.fit_params
        self.actual_fparams.update(fit_params)

        alg = self.actual_fparams['alg']
        maxits = self.actual_fparams['maxits']
        ftol = self.actual_fparams['ftol']
        xtol = self.actual_fparams['xtol']
        Lnorm = self.actual_fparams['Lnorm']

        Te = init['Te']
        Vf = init['Vf']
        I0 = init['I0']
        if I0 is None:
            I0 = 1.7 * np.average(np.clip(self.i, 0, 1000))

        var = [Te, Vf, I0]
        scale = np.array([Te, Te/3, I0])
        if alg == 'leastsq':
            pfit, pcov, infodict, msg, ier = \
                leastsq(self.residuals, var, args=(self.v, self.i), maxfev=maxits,
                        full_output=1, diag=scale, ftol=ftol*I0, xtol=xtol)
            # residual already raised to power Lnorm/2 = so just square here
            resid = np.power(np.mean(np.abs(infodict['fvec']**2)),1./Lnorm)
            # this is required to scale the covariance
            s_sq = (self.residuals(pfit, self.v, self.i)**2).sum()/(len(self.i)-len(init))
            fit_results = [pfit, resid, infodict['nfev']]
            if ier not in [1,2,3,4]: 
                fit_results[-1] = 9000+ier  # keep it in uint16 range
            if self.parent is not None and self.parent.debug>0:
                if self.parent.debug>1 or ier not in [1,2,3,4]:
                    print('fit:', ier, msg)

        elif alg == 'amoeba':
            fit_results = list(amoeba.amoeba(var, scale, self.error_fun, 
                                             itmax=maxits, ftolerance=ftol*I0,
                                             xtolerance=xtol,
                                             data=dict(v=self.v, i=self.i)))
            fit_results[1] = -fit_results[1]  # change residual back to positive
            s_sq = None  # don't know how to get cov for amoeba
            pcov = None

        Te, Vf, I0 = fit_results[0]
        fit_results[1] /= I0  # normalise to IO 
        residual, mits = fit_results[1:3]
        fit_results.append(maxits)
        self.pcov = pcov
        self.s_sq = s_sq

        if plot > 1:
            plt.scatter(self.v, self.i, c='r', s=80,
                        alpha=min(1, 4/np.sqrt(len(self.v))))

            self.plotchar(self.v, Te, Vf, I0, linewidth=4)
            # provide brief info about algorithm and Lnorm in the title
            plt.title('Te = {Te:.1f}eV, Vf = {Vf:.1f}, Isat = {I0:.3g}, <res> = {r:.2e} {mits} {A}{p}:its '
                      .format(Te=Te, Vf=Vf, I0=I0, r=float(residual), mits=mits, A=alg.upper()[0],p=Lnorm))
            plt.show(block=0)

        return(fit_results)


def tog(colls=None, hide_all=0):
    """ toggle lines on and off, unless hide_all=1 """
    if colls is None:
        colls = []
        for c in plt.gca().get_children():
            if type(c) == plt.matplotlib.lines.Line2D:
                colls.append(c)

    for coll in colls:
        if hide_all:
            coll.set_visible(0)
        else:
            coll.set_visible(not coll.get_visible())

    plt.show()


def tryone(var, data, mrk='r', w=4):
    """ evaluate fit and show curve - meant for manual use
    tryone([Te,Vf,I0],dict(v=sweepV,i=iprobe))
    """
    # tryone([25,-15, .02],dict(i=iprobe, v=sweepV))
    print('-err fun', -error_fun(var, data))
    Te, Vf, I0 = var
    v = data['v']
    i = data['i']
    plotchar(v, Te, Vf, I0, linewidth=w)


def find_sat_level(i_probe, debug=1):
    """ Initially written to find the saturation of electrons, but will work for ions 
    if we flip the sign
    If this is not found, return None
    """
    if debug:
        print('find sat level')
    label = 'find_sat level'
    try:  # put the whole thing in try/except - should really try to track errors
        cnts,bins = np.histogram(i_probe, bins=20)
        wneg = np.where(bins < 0)[0]
        wneg = wneg[:-2]  # chop out the last one - the one closest to zero in case it has some isat
        maxnegidx = np.argmax(cnts[wneg])
        if cnts[maxnegidx] < 5: # peak is too low to expect a meaningful result 
            return(None)
        wnegsml = np.where((wneg > maxnegidx) & (cnts[wneg] < cnts[wneg[maxnegidx]]/3.))[0]
        if debug > 1:
            plt.figure()
            plt.hist(i_probe, bins=20)
            plt.plot(bins[wneg[wnegsml]], cnts[wneg[wnegsml]],'ro')
            plt.title('analysis of ' + label.replace('(','<'))
        # the first bin in this 'smaller' current set (1 +  gives the RhS of it)
        #try:
        cutoff = bins[1 + wneg[wnegsml]][0]
    #except IndexError as reason:
    except Exception as reason:
        if debug > 0:
            print('Error in wned/wnegsml lookup ' + str(reason))
        return(None)
    return(cutoff, np.nan)


    return None

def find_clipped(sigs, clipfact):
    """ look for digitizer or amplifier saturation in all raw
    signals, and return offending indices.  Detecting digitizer saturation
    is easy, amplifier saturation is softer - need to be careful

    sigs can be a signal, array or a list of those
    Note that restore_sin() can be used on the sweep, rather than omitting
    the points.
    """
    if isinstance(sigs, (list, tuple)):
        goodbits = np.ones(len(sigs[0]), dtype=np.bool)
        for sig in sigs:
            goodbits = goodbits & find_clipped(sig, clipfact)
            debug_(debug, 5, key='find_clipped')

        return(goodbits)

    cnts, vals = np.histogram(sigs, 50)   # about 100ms for 250k points

    if (np.sum(cnts[0]) > clipfact * np.sum(cnts[1])):
        wunder = (sigs < vals[1])
    else:
        wunder = np.zeros(len(sigs), dtype=np.bool)

    if (np.sum(cnts[-1]) > clipfact * np.sum(cnts[-2])):
        wover = (sigs > vals[-2])
    else:
        wover = np.zeros(len(sigs), dtype=np.bool)

    return(~wunder & ~wover)



class Langmuir_data():
    """ get the fits for a multi (or single) cpt diagnostic using segments
    If a dictionary params is supplied, use these to process the data, 
    otherwise just read in the data.

    Eternal question: Should I pass parameters or set them in the object?
    Answer for now - if you want them in the record of analysis, pass 
    them through process_swept_Langmuir

    obvious candidates for object: debug, plot, select
    obvious candidates for args: Lnorm, initial_TeVfI0, clipfact, rest_swp

    If params contains a filename entry, save the result as that name

    Final solution: save the ACTUAL params used in the object, and then 
    save THOSE in when writing, not the onbes entered.

    See :py:meth:`process_swept_Langmuir` for the processing arguments
    """
    def __init__(self, shot, i_diag, v_diag, dev_name="W7X", debug=debug, plot=1, verbose=0, params=None, time_range=None):
        """
Create a Langmuir Data object for later processing

Args:
        shot: shot number
        i_diag: name of probe corrent diagnostic - at present should be a multi channel
        v_diag: sweep voltage diagnostic - should include all voltages referred to by all the corrent channels config as sweepV
        dev_name:
        debug:
        plot:  can be set later in process_langmuir
        params: if set, can cause the process step to be executed after loading
"""
        if pyfusion.NSAMPLES != 0:
            raise ValueError("doesn't make sense to use minmax decimation here")
        self.dev = pyfusion.getDevice(dev_name)
        self.shot = shot
        self.verbose = verbose
        self.i_diag = i_diag
        self.v_diag = v_diag
        self.debug = debug
        self.plot = plot
        self.select = None
        self.t_comp = (0.1,0.2)
        self.params = params
        self.figs = []
        self.suffix = ''  # this gets put at the end of the fig name (title bar)

        self.imeasfull = self.dev.acq.getdata(shot, i_diag, contin=True, time_range=time_range)
        self.vmeasfull = self.dev.acq.getdata(shot, v_diag, contin=True, time_range=time_range)
        comlen = min(len(self.vmeasfull.timebase), len(self.imeasfull.timebase))
        FFT_size = nice_FFT_size(comlen-2, -1)
        # the minus 2 is a fudge to hide small inconsistencies in reduce_time
        # e.g. 20160310 9 W7X_L5_LPALLI
        self.imeasfull = self.imeasfull.reduce_time([self.imeasfull.timebase[0], self.imeasfull.timebase[FFT_size]])
        self.vmeasfull = self.vmeasfull.reduce_time([self.vmeasfull.timebase[0], self.vmeasfull.timebase[FFT_size]])

        if self.params is not None:
            self.process_swept_Langmuir(**self.params)


    def get_iprobe(self, leakage=None, t_comp=None):
        """ The main purpose is to subtract leakage currents
        Will use the full data, as the t_range is meant to be plasma interval
        returns (by setting self.iprobefull) a copy of the measured current, 
        overwritten with the corrected current iprobe (so it stays a signal)

        sets self.iprobefull, self.sweepQ   and also appends to self.comp
        """
        # obtain leakage estimate
        if t_comp is None:
            t_comp = self.t_comp
        FFT_size = nice_FFT_size(len(self.imeasfull.timebase), -1)
        self.iprobefull = self.imeasfull.copy()
        self.sweepQ = []  # will keep these for synchronous sampling
        input_leakage = leakage
        for (c, chan) in enumerate(self.imeasfull.channels):
            if np.all(t_comp == [0,0]): # skip compensation,AND DC correction - not so good
                return
            if self.select is not None and c not in self.select:
                continue
            leakage = input_leakage
            cname = chan.config_name
            sweepV = self.vcorrfull[self.vassoc[c]][0:FFT_size]
            sweepQ = hilbert(sweepV)
            self.sweepQ.append(sweepQ)  # save for synchronising segments (it is smoothed)

            # these attempts to make it accept a single channel are only partial
            imeas = self.imeasfull.signal[c] # len(self.imeasfull.channels) >1 else self.imeasfull.signal
            tb = self.imeasfull.timebase

            w_comp = np.where((tb>=t_comp[0]) & (tb<=t_comp[1]))[0]
            if len(w_comp) < 2000:

                raise ValueError('Not enough points {wc} in t_comp for leakage est. - try {tt}'
                    .format(tt=np.round([tb[0], tb[0] + t_comp[1]-t_comp[0]],3),
                            wc=len(w_comp)))
            ns = len(w_comp)
            wind = np.blackman(ns)
            offset = np.mean(wind * imeas[w_comp])/np.mean(wind)
            # Voffset is only used later in deducing cross-talk at DC
            Voffset = np.mean(wind * sweepV[w_comp])/np.mean(wind)
            sweepVFT = np.fft.fft(AC(sweepV[w_comp]) * wind)
            imeasFT = np.fft.fft(AC(imeas[w_comp]) * wind)
            ipk = np.argmax(np.abs(sweepVFT)[0:ns//2])  # avoid the upper peak
            # ipk is the index of the FFT max
            comp = imeasFT[ipk]/sweepVFT[ipk]
            # Try to detect bad choice of compensation region
            harmonics = np.sqrt(np.linalg.norm(imeasFT[3*ipk//2:6*ipk])) / np.abs(imeasFT[ipk])
            print('Harmonics = ', harmonics)
            # Note that if temperature is high, the content is small and may not show
            #  See for example LP10 in:
            # run  pyfusion/examples/run_process_LP.py  sweep_sig='W7X_L57_LP01_U' replace_kw='dict(t_range=[.0,1.1],plot=0,t_comp=[1.08,1.1])' shot_list='[[20160309,13]]' seglist=[7] lpdiag='W7X_L57_LPALLI'

            if harmonics > 0.1:
                pyfusion.logging.warn('Harmonics large in t_comp region')
                pyfusion.utils.warn('Harmonics large in t_comp region')
            if self.debug>2:
                plt.figure()
                plt.plot(np.fft.fftfreq(ns, np.diff(tb)[0] ), np.abs(imeasFT))
                plt.title('{ch}: {ha:.3f}'.format(ch=chan.name, ha=harmonics))
                if harmonics > 0.1:
                    plt.figure()
                    plt.plot(tb[w_comp], imeas[w_comp])

            debug_(pyfusion.DEBUG, 1, key='check_comp_harms')
            self.comp.append(comp)
            #print('leakage compensation factor = {r:.2e} + j{i:.2e}'
            #      .format(r=np.real(comp), i=np.imag(comp)))
            print('{ch}: DC {DCl:9.2e}, {u}sing the computed leakage conductance = {m:.2e} e^{p:5.2f}j'
                  .format(u = ["Not u", "U"][leakage is None], ch=chan.name,
                          m=np.abs(comp), p=np.angle(comp), DCl=float(offset/Voffset)))
            if leakage is None:
                leakage = [np.real(comp), np.imag(comp)]
            elif np.isscalar(leakage):
                leakage = [np.real(leakage), np.imag(leakage)]
            else:
                leakage = [np.real(leakage[c]), np.imag(leakage[c])]

            # find the common length - assuming they start at the same time????
            comlen = min(len(self.imeasfull.timebase),len(self.vmeasfull.timebase),len(sweepQ))
            # put signals back into rdata (original was copied by reduce_time)
            # overwrite - is this OK?
            self.iprobefull.signal[c] = self.iprobefull.signal[c]*0.  # clear it
            # sweepV has a DC component! that should not be removed - beware!
            self.iprobefull.signal[c][0:comlen] = self.imeasfull.signal[c][0:comlen]-offset \
                                        - sweepV[0:comlen] * leakage[0] - sweepQ[0:comlen] * leakage[1]
            # remove DC cpt from I (including that just added from the compensation sweepV)
            offset = np.mean(wind * self.iprobefull.signal[c][w_comp])/np.mean(wind)
            self.iprobefull.signal[c][0:comlen] -= offset

    def prepare_sweeps(self, rest_swp='auto', sweep_freq=500, t_offs=3, Vpp=90*2, clip_level_minus=-88):
        """ extracts sweep voltage data for all probes,
        call restore_sin if necessary
        Initially, the voltage data is assumed to be in a dictionary
        """
        if self.debug > 0:
            print('entering prepare_sweeps ', len(self.imeasfull.signal[0]))
        if str(rest_swp).lower() == 'auto':
            # bug: was wrong 0309,0 - but didn't affect any data I sent
            rest_swp = self.shot > [20160310, 0]  and  self.shot < [20160310, 99] 
            print ('* Automatically setting rest_swp to {r} *'.format(r=rest_swp))

        self.vcorrfull = self.vmeasfull.copy()
        if t_offs is not None and t_offs is not 0:
            self.vcorrfull.signal[t_offs:] = self.vcorrfull.signal[:-t_offs]
        if rest_swp:  # last minute import reduces dependencies for those who don't need it
            from pyfusion.data.restore_sin import restore_sin
            for (c, sig) in enumerate(self.vmeasfull.signal):
                self.vcorrfull.signal[c] = restore_sin(self.vmeasfull, chan=c, sweep_freq=sweep_freq,
                                                       Vpp=Vpp, clip_level_minus=clip_level_minus, verbose=self.verbose)

    def fit_swept_Langmuir_seg_multi(self, m_seg, i_seg, v_seg, clipfact=5,  initial_TeVfI0=None, fit_params=None, plot=None):
        if len(v_seg.timebase) != len(i_seg.timebase):
            pyfusion.logging.warn('Unequal timebases {vl}, {il}'.format(vl=len(v_seg.timebase), il=len(i_seg.timebase)))
            debug_(self.debug, 1, key='Unequal_timebases')
            return(None)
        if self.select is None:
            self.select = range(len(self.i_chans))
        res = []
        debug_(self.debug, 2, key='fit_swept_multi')
        for (c, chan) in enumerate(i_seg.channels):
            if self.select is not None and c not in self.select:
                continue
            v = v_seg[self.vassoc[c]]
            i = i_seg.signal[c]
            im = m_seg.signal[c]
            result = self.fit_swept_Langmuir_seg_chan(im, i, v, i_seg, channame=chan.config_name, clipfact=clipfact,  initial_TeVfI0=initial_TeVfI0, fit_params=fit_params, plot=plot)
            if fit_params.get('esterr', None) is not None:             
                [Te, ne, I0], resid, nits, maxits, Ie_Ii, errest = result
                eTe, ene, eI0 = errest
            else:
                [Te, ne, I0], resid, nits, maxits, Ie_Ii = result

            t_mid = np.average(i_seg.timebase)
            res_list = [t_mid, Te, ne, I0, resid, nits, maxits, Ie_Ii]
            if fit_params.get('esterr', None) is not None:
                res_list.extend( [eTe, ene, eI0])
            res.append(res_list)
            if len(res) < 1:
                pyfusion.utils.warn('No segments analysed - is select OK')
        return(res)
    
    def fit_swept_Langmuir_seg_chan(self, m_sig, i_sig, v_sig, i_segds, channame, fit_params=None, clipfact=5,  initial_TeVfI0=None, plot=None):
        """Lowish level routine - takes care of i clipping and but not
        leakage or restoring sweep.

        m_seg is the segment of measured current before compensation (for clip checks)
        Returns fitted params including error
    """
        # Note that we check the RAW current for clipping
        try:  # this relative import doesn't work when process_Lang is run (for test)
            from pyfusion.data.LPextra import lpfilter, estimate_error
        except ValueError:
            from pyfusion.data.LPextra import lpfilter, estimate_error

        lpf = fit_params.get('lpf', None)
        cycavg = fit_params.get('cycavg', None)
        minleft = fit_params.get('minleft', [0.3, 20])  # min fraction remaining, min num remaining

        segtb = i_segds.timebase
        v, i, im = v_sig.copy(), i_sig.copy(), m_sig.copy()     # for fit
        # v_n, i_n, im_n = v * np.nan, i * np.nan, im * np.nan,   # for nan plots

        good = find_clipped([v, im], clipfact=clipfact)
        wg = np.where(good)[0]
        if (~good).any():  # if any bad
            if len(wg) < minleft[0] * len(v) or len(wg) < minleft[1]:  # was (~good).all()
                print('{t:.5f}s suppressing all !*** {a} *************************8'
                      .format(a=len(wg),t=np.average(segtb)))
                if fit_params.get('esterr', None) is not None:
                    return([3*[None], None, None, None, None, 3*[None]]) # incl dummy errsa
                else:
                    return([3*[None], None, None, None, None])

            elif self.verbose > 0 or len(wg) < 0.9 * len(v):
                print('{t:.5f}s suppress clipped {b}/{a}'
                      .format(b=len(np.where(~good)[0]), a=len(good), t=np.average(segtb)))

        # [[Te, Vf, I0], res, its]
        if plot is None:
            # plot in detail only if there are 8 figures or less
            fmax = 8 if os.name == 'posix' else 20  # windows doesn't slow down much
            # should consider plt.get_fignums()
            plot = self.plot + 2*(fmax > (len(list(self.segs))*len(self.imeasfull.channels)))
        if (plot >= 2) and len(self.figs) < 8:  # 20 really slows down
            # best to open in another session.
            interval = str('{t}'.format(t=[round(segtb[0], 6), round(segtb[-1], 6)]))
            titbar ='{i} {c}{s}'.format(i=interval, c=channame, s=self.actual_params['suffix'])
            if plot == 4:  # 4 suppresses bar to allow many - self.plot=4 gets promoted to 6 here
                titbar = None
            if self.debug > 0: print('plot={p}'.format(p=plot),  end=': ')
            fig, axs = plt.subplots(nrows=1 + (plot >= 3), ncols=1, squeeze=0, num=titbar)

            self.figs.append(fig)
            if plot >= 3:
                # these labels help find the data
                axs[1, 0].plot(segtb, i, 'b', lw=.2, label='i_probe')
                axs[1, 0].plot(segtb[wg], i[wg], 'b.')
                axs[1, 0].set_ylabel('i_probe', color='b')
                axv = axs[1, 0].twinx()
                ylab = 'sweep V'
                if cycavg is not None and cycavg[2] is not 0:
                    ylab += ' (no shift)'
                axv.set_ylabel(ylab, color='r')
                axv.plot(segtb, v, 'r', lw=.2, label='v')
                axv.plot(segtb[wg], v[wg], 'r.')
                plt.sca(axs[1, 0])  # ready to be overlaid by filtered I

            fig.suptitle('{shot} {tr} {d}:{c}'
                         .format(tr=interval, shot=self.shot, d=self.i_diag, c=channame))

        #res = LPfit(v, i, plot=(debug > 1),maxits=100)
        if lpf is not None:
            # We filter i only because our Vs are usually clean enough
            i = lpfilter(segtb, i, v, lpf=lpf, debug=self.debug, plot=self.plot)
            # check if i has been truncated to an even number by real fft - 
            if wg[-1] >= len(i):  # if so, truncate the other pieces
                wg = wg[0:-1]
                v = v[0:-1]
                segtb = segtb[0:-1]
        if cycavg is not None:
            i = boxcar(i, period=cycavg[0], maxnum=cycavg[1], debug=debug)
            v = boxcar(v, period=cycavg[0], maxnum=cycavg[1])
            v = rotate(v, offs=cycavg[2])
            # for simplicity use boxcar (max=1) to get the right segtb size and offset
            segtb = boxcar(segtb, period=cycavg[0], maxnum=1)
            # need to re-do wg as any of the averaged cycles could have forced a nan
            wg = np.where(~np.isnan(i))[0]
            
        if (plot >= 2) and len(self.figs) < 8:
            plt.sca(axs[0, 0])  # ready for VI curve

        self.fitter = LPfitter(i=i[wg], v=v[wg], fit_params=fit_params, plot=plot, parent = self)
        result = self.fitter.fit(init=initial_TeVfI0, fit_params=fit_params, plot=plot)
        # get a 'fitted' value of most negative current to give us Ie_max
        # Use the fitted curve evaluated at the max voltage (measured V is smooth)
        maxv = np.max(v)
        debug_(self.debug, 2, key='after_fit')
        Te, Vf, I0 = result[0]
        Imaxsmoothed = self.fitter.LPchar(maxv, Te, Vf, I0)
        Ie_Ii = (I0-Imaxsmoothed)/I0
        result.append(Ie_Ii)
        #  print(Te, result)
        if fit_params.get('esterr', None) is not None:
            #if fit_params['esterr'] != 'cov':
            result.append(estimate_error(self, result, debug=self.debug, method=fit_params['esterr']))
        return(result)

    def write_DA(self, filename):
        from pyfusion.data.DA_datamining import DA,  Masked_DA
        dd = {}
        res = np.array(self.fitdata, dtype=np.float32)
        nt = len(res)
        nc = len(res[0])
        for key in ['date', 'progId', 'shot']:
            dd[key] = np.zeros(nt, dtype=np.int64)
            
        dd['date'][:] = self.shot[0]
        dd['progId'][:] = self.shot[1]
        dd['shot'][:] = self.shot[1] + 1000*self.shot[0]

        for key in ['nits','maxits']:
            dd[key] = np.zeros([nt,nc], dtype=np.uint16)

        # make all the f32 arrays - note - ne is just I0 for now - fixed below
        # lookup is a list to help sort out the list of results from the fit
        lookup = [(0, 't_mid'), (1, 'Te'), (2, 'Vf'), (3, 'I0'), 
                  (4, 'resid'), (5, 'nits'), (6, 'maxits'), (7, 'Ie_Ii'),
                  (3, 'ne18')]

        if self.fitter.fit_params.get('esterr', None) is not None:
            lookup.extend([(8, 'eTe'), (9, 'eVf'), (10, 'eI0') ])

         # clip_iprobe should be a fit_params so that it can work at the seg_chan level
        clip_iprobe = self.actual_params['clip_iprobe']
        if len(self.fitdata[-1]) == 1 and clip_iprobe is not None and len(np.shape(clip_iprobe)) == 0:
            lookup.extend([(1+np.max([l[0] for l in lookup]), 'esat_clip')])
        if len(np.unique([l[0] for l in lookup])) != np.shape(self.fitdata[0])[1]:
            debug_(0, 0, key='process_loop')

        for (ind, key) in lookup:
            if key not in dd:
                dd[key] = np.zeros([nt, nc], dtype=np.float32)
            dd[key][:] = res[:, :, ind]

        # fudge t_mid is not a vector...should fix properly
        dd['t_mid'] = dd['t_mid'][:, 0]
        try:
            import getpass
            username = getpass.getuser()
        except:
            if hasattr(os, 'getlogin'):
                username = os.getlogin()
            else:
                username='?'
                
        dd['info'] = dict(params=self.actual_params,
                          coords=[self.coords[ic] for ic in self.select],
                          #area=[self.area[ic] for ic in self.select], # needs to be in npz file etc first
                          shotdata=dict(shot=[self.shot], utc_ns=[self.imeas.utc[0]]),
                          channels=[chn.replace(self.dev.name+'_', '')
                                    .replace('_I', '')
                                    for chn in
                                    [self.i_chans[ic] for ic in self.select]],
                          orig_name = os.path.split(filename)[-1],
                          username = username)
        
        da = DA(dd)
        da.masked = Masked_DA(['Te', 'I0', 'Vf', 'ne18', 'Ie_Ii'], baseDA=da)
        #  da.da['mask']=(da['resid']/abs(da['I0']) < .7) & (da['nits']<100)
        #  da.da['mask'] = ((da['resid']/abs(da['I0']) < .7) & (da['nits'] < da['maxits'])
        # from version 0.7.0 onwards, resid is already normed to I0
        lpf = self.fitter.actual_fparams['lpf']
        # Note: these multilines ('down to here') can be applied to a DA by 
        #       pasting to reset mask AFTER uncommenting the following # line
        # lpf = da['info']['params']['actual_fit_params']['lpf']
        rthr = 0.7  # LP20160309_29_L53__amoebaNone1.2N_2k.npz is < .12  others 
                    # None 0310_9 up to 0.7-0.8
        if lpf is not None:
            rthr = rthr * np.sqrt(lpf/100.0)
        da.da['mask'] = ((da['resid'] < rthr) & (da['nits'] < da['maxits'])
                         & (np.abs(da['Vf']) < 200) & (np.abs(da['Te']) < 200) 
                         & (da['I0']>0.0004))
        # additional restriction applied if the error estimate is available
        if 'eTe' in da.da:  # want error not too big and smaller than temp
            da.da['mask'] &= ((np.abs(da['eTe']) < 100)
                              & (np.abs(da['eTe']) < np.abs(da['Te'])))
        #   down to here
        qe = 1.602e-19
        mp = 1.67e-27
        fact = 1/(0.6*qe)*np.sqrt(self.amu*mp/(qe))/1e18         # units of 1e18
        # check if each channel has an area

        for (c, chn) in enumerate([self.i_chans[ic] for ic in self.select]):
            cd = get_config_as_dict('Diagnostic', chn)
            A = cd.get('area', None)
            if A is None:
                A = 1.0e-6
                pyfusion.logging.warn('Defaulting area for {chn} to {A}'.format(chn=chn, A=A))
            A = float(A)
            da.da['ne18'][:, c] = fact/A * da['I0'][:, c]/np.sqrt(da['Te'][:, c])
        da.save(filename)

    def process_swept_Langmuir(self, t_range=None, t_comp=[0, 0.1], fit_params = dict(maxits=200, alg='leastsq',esterr='cov', lpf=None), initial_TeVfI0=dict(Te=50, Vf=15, I0=None), dtseg=4e-3, overlap=1, rest_swp='auto', clipfact=5, clip_iprobe = None, clip_vprobe=None, leakage=None, threshold=0.01, threshchan=12, filename=None, amu=1, plot=None, plot_DA=False, return_data=False, sweep_freq=500, t_offs=0, defaults=dict(sweep_freq=500, dtseg=4e-3), suffix='', debug=None):
        """ 
Process the I, V probe data in the Langmuir_data object

Args:
  fit_params: a dictionary of parameters used in the fitting stage (passed to the  LP_fitter class)
    Most of these have reasonable defaults (for W7X limiter probes), and the *actual* values used are 
    attached to the Langmuir_data object and saved in the the DA_ file.
  filename: The output (DA) file name, defaults to a pattern containing the shot number etc.  If a user 
    entered value contains a '*', then that pattern is inserted in its place e.g. filename='test_*' 
    generates a DA file test_LP20160308_23_L53.npz.
  t_range: time range in secs over which data is processed - None uses a minimum current value as follows.
  threshold: processing starts once this value is exceeded in the following channel (assuming t_range=None).
  threshchan: the current channel number (base 0) used to detect plasma start.
  overlap: degree of overlap of the analysed segments: 1 is no overlap, 2 means half the segment is common
    to neighboring data.
  t_offs: shift of timebase in samples to allow for bridge probe - default 0
  dtseg: The length of a segment over shich analysis is performed, either in seconds (float) or samples (int)
  rest_swp: Restore clipped sweep V if True.  If None, for March 10 only.
  leakage: A complex conductance to represent the crosstalk between voltage and current channels.  If None
    it is automatically calculated for the interval t_comp.  To set to zero, use [0,0] 
clip_iprobe = e.g. [-0.015, .02]  # used to check if a resistive term is affecting Te
clip_vprobe = [,]  # used to reduce emphasis on isat
  debug = [None - use the existing value]
  plot_DA - plot the DA in figure(plot_DA) (and upwards?)

Returns:
    (if return_results) A list of results - can save a DA object as described above if a filename is given

Raises:
    ValueError:



==> results[time,probe,quantity]
plot = 1   : V-I and time data but only if there are not too many.
plot >= 2  : V-I curves
plot >= 3  : ditto + time plot
plot >= 4  : ditto + all I-V iterations
  ** note - value will remain once set.

Can send parameters through the init or process - either way they are all 
   recorded in actual_params


Start by processing in the ideal order: - logical, but processes more data than necessary
fix up the voltage sweep
compensate I
detect plasma
reduce time
segment and process

Faster order - reduces time range earlier: (maybe implement later)
detect plasma time range with quick and dirty method
restrict time range, but keep pre-shot data accessible for evaluation of success of removal
        """
        if debug is not None:
            self.debug = debug
        if plot is not None:
            self.plot = plot
        self.figs = []  # reset the count of figures used to stop too many plots
        self.actual_params = locals().copy()
        # try to catch mispelling, variables in the wrong place
        for k in fit_params:
            if k not in 'Lnorm,cov,esterr,alg,xtol,ftol,lpf,maxits,track_ratio,cycavg,minleft'.split(','):
                raise ValueError('Unknown fit_params key ' + k)

        if (fit_params.get('cycavg', None) is not None and
            clip_iprobe is not None):
            raise ValueError("Can't use clip_iprobe with cycavg at present")

        self.actual_params.pop('self')
        self.actual_params.update(dict(i_diag=self.i_diag, v_diag=self.v_diag))
        # and do it for actuals too
        for k in self.actual_params:
            if k not in 'amu,clipfact,clip_iprobe,clip_vprobe,dtseg,filename,initial_TeVfI0,leakage,overlap,plot,plot_DA,rest_swp,suffix,t_comp,t_range,t_offs,threshold,threshchan,fit_params,v_diag,i_diag,return_data,sweep_freq,defaults,debug'.split(','):
                raise ValueError('Unknown actual_params key ' + k)

        self.actual_params.update(dict(i_diag_utc=self.imeasfull.utc, pyfusion_version=pyfusion.VERSION))
        self.amu = amu
        t_offs = 3 if t_offs is None and 'BRIDGE' in self.i_diag else t_offs
        if t_offs != 0 and 'cycavg' in fit_params:
            wait_for_confirmation('Do you really want t_offs = {t_offs} and cycavg = {cycavg}'
                                  .format(t_offs=t_offs, cycavg=fit_params['cycavg']))
        if not isinstance(self.imeasfull.channels, (list, tuple, np.ndarray)):
            self.imeasfull.channels = [self.imeasfull.channels]

        self.i_chans = [ch.config_name for ch in self.imeasfull.channels]
        # A multi diag may contain voltage channels - let's discard them
        for ch in self.i_chans:  # only take the current channels, not U
            if self.dev.name.startswith == 'W7' and ch[-1] != 'I':
                raise ValueError("Warning - removal of V chans doesn't work!!!")
                # hopefully we can ignore _U channels eventually, but not yet
                self.i_chans.remove(ch)

        self.coords = [ch.coords.w7_x_koord for ch in self.imeasfull.channels] if self.dev.name.startswith('W7') else []


        # if you want just one voltage channel, at the moment, you still
        # need it to be a multi-channel diag. (to simplify this code).
        self.v_chans = [ch for ch in self.vmeasfull.keys()]  # this works for single and multi diags

        # want to say self.vmeas[ch].signal where ch is the imeas channel name
        self.vlookup = {}
        for (c, vch) in  enumerate(self.v_chans):
            self.vlookup[vch] = c

        self.vassoc = []    # list of sweepVs associated with each i channel
        # one per i channel - these refer to their
        # respective self.v_chans
        # for OP1.1, only a few V chans were recorded and
        # in practice only two channels are necessary.

        default_sweep = self.v_chans[0] if len(self.v_chans) == 1 else 'NO SWEEP'
        if np.isscalar(self.shot):
            compfun = int
        else:
            compfun = tuple
        if default_sweep is 'NO SWEEP':

            if compfun(self.shot) > compfun([20171231, 999]):
                default_sweep = 'W7X_PSUP2_U'  # 
            elif compfun(self.shot) > compfun([20170926, 999]):
                default_sweep = 'W7X_KEPCO_U'  # only really working after shot 50ish
                # default_sweep = 'W7X_UTDU_LP18_U'  # only really working after shot 50ish
            elif compfun(self.shot) > compfun([20160310, 999]):
                default_sweep = 'W7X_LTDU_LP18_U'  # 13 is dead up to 0921 at least
            else:
                default_sweep = 'W7X_L57_LP01_U'

        for ch in self.i_chans: # find the matching index to the multichannel diag (have I done this earlier?)
            mcdnum = [n for n in range(len(self.imeasfull.channels)) if self.imeasfull.channels[n].name == ch]
            if len(mcdnum) != 1:
                raise LookupError('Channel {ch} not in {i_diag}'.format(ch=ch, i_diag=i_diag))
            if hasattr(self.imeasfull.channels[mcdnum[0]], 'vsweep'):
                vsweep_diag = self.imeasfull.channels[mcdnum[0]].vsweep
            else:
                cd = get_config_as_dict('Diagnostic', ch)
                # TODO(bdb): use of default_sweep should generate a warning
                vsweep_diag = cd.get('sweepv', default_sweep)
            self.vassoc.append(vsweep_diag)

        debug_(self.debug, 2, key='after get associated')
        # first do the things that are better done on the whole data set.
        # prepare_sweeps will populate self.vcorrfull
        # self.check_crosstalk(verbose=0)  # this could be slow

        # Check that the substitutions in the filename won't cause an error
        # at the last minute so it can be fixed BEFORE a long computation.
        parm_dict = dict(s0=self.shot[0], s1=self.shot[1], i_diag=self.i_diag,
                         t_start=t_range[0], dtseg=dtseg)

        if filename is not None and '}' in filename:
            try:
                testname = filename.format(**parm_dict)
            except KeyError as reason:
                print(str(reason), ' filename variables are ',
                      str(parm_dict).replace(', ','\n'))
                raise ValueError('Unknown filename variable {v}: known variables are '
                                 + ', '.join(list(parm_dict))
                                 .format(v=str(reason)))
            
        # begin crunching 
        self.prepare_sweeps(rest_swp=rest_swp, sweep_freq=sweep_freq, t_offs=t_offs)
        self.comp = []
        self.get_iprobe(leakage=leakage, t_comp=t_comp)

        tb = self.iprobefull.timebase
        # the 3000 below tries to avoid glitches from Hilbert at both ends
        #w_plasma = np.where((np.abs(self.iprobefull.signal[threshchan]) > threshold) & (tb > tb[3000]) &(tb < tb[-3000]))[0]
        # only look at electron current - bigger (shot 0309.52 LP53 has a positive spike at 2s)
        
        if t_range is None:
            w_plasma = np.where((-self.iprobefull.signal[threshchan] > threshold) &
                                (tb > tb[3000]) &(tb < tb[-3000]))[0]
            t_range = [tb[w_plasma[0]], tb[w_plasma[-1]]]

        self.t_range = t_range
        if t_range is not None:
            self.imeas = self.imeasfull.reduce_time(t_range)
            self.iprobe = self.iprobefull.reduce_time(t_range)
            self.vcorr = self.vcorrfull.reduce_time(t_range)
        else:
            self.imeas = self.imeasfull
            self.iprobe = self.iprobefull
            self.vcorr = self.vcorrfull

        # We need to segment iprobe, also i_meas to check clipping, and vsweep
        self.segs = zip(self.imeas.segment(dtseg, overlap),
                        self.iprobe.segment(dtseg, overlap),
                        self.vcorr.segment(dtseg, overlap))

        if self.debug>0: print('Use {n} segments'.format(n=len(self.segs)))
        self.fitdata = []
        debug_(self.debug, 3, key='process_loop')
        for imseg, iseg, vseg in self.segs:
            # print('len vseg', len(vseg.signal[0]))
            if len(vseg.timebase) < dtseg:  # //5:  # skip short segments - was //5 why?
                continue

            print(np.round(np.mean(imseg.timebase),4), end='s: ')  # midpoint    
            esat_clip = None
            if clip_iprobe is not None:
                if len(np.shape(clip_iprobe)) == 0: # this means we want saturation clipping
                    this_clip_iprobe = find_sat_level(iseg.signal[0], debug=self.debug)
                    print('Found saturation level of ', this_clip_iprobe)
                    if this_clip_iprobe is None:
                        #break doesn't work - really need to drop two out
                        this_clip_iprobe = [np.nan,np.nan]
                        esat_clip = this_clip_iprobe[0]

                else:
                    print('fudge hard I clipping', end=' ')
                    # This should only be used to simulate problems.
                    # However it usually works to force points outside the
                    # range to be ignored - but not always - esp. with cycavg
                    this_clip_iprobe = clip_iprobe
                    
                # have to clip the raw signal, because that is where the decision is made
                # but we have to do it AFTER iseg is extracted, because the leakage
                # will make the clipping uneven (a sinusoidal top) - so clip iseg too
                iseg.signal = np.clip(iseg.signal, *this_clip_iprobe)
                imseg.signal = iseg.signal.copy()  # isn't this in the wrong order?

            if clip_vprobe is not None: # fudge hard clipping - only for simulating problems
                print('fudge hard V clipping', end=' ')
                vseg.signal = np.clip(vseg.signal, *clip_vprobe)
                #vmseg.signal = vseg.signal.copy()  # need to implement, but mainly for graphics
                vmseg = vseg.copy()  # need to implement, but mainly for graphics

            self.fitdata.append(self.fit_swept_Langmuir_seg_multi(imseg, iseg, vseg, clipfact=clipfact, initial_TeVfI0=initial_TeVfI0, fit_params=fit_params, plot=plot))
            #self.fitdata[-1].append(dict(esat_clip = esat_clip))
            #if the record lengths are different None is returned - spit it out.
            if self.fitdata[-1] is None:
                self.fitdata.pop()
            # beware - the esat_clip is a channel thing - needs to be inside!
            if len(self.fitdata[-1]) == 1 and clip_iprobe is not None and len(np.shape(clip_iprobe)) == 0:
                self.fitdata[-1][0].append(esat_clip)
        # note: fitter.actual_fparams only records the most recent!
        
        self.actual_params.pop('fit_params') # only want the ACTUAL ones.
        self.actual_params['actual_fit_params'] = self.fitter.actual_fparams
        if filename is not None:            
            if  '*' in filename:
                fmt = 'LP{s0}_{s1}_'
                if 'L5' in self.i_diag:
                    fmt += 'L5' + self.i_diag.split('L5')[1][0]
                elif 'TDU' in self.i_diag:
                    fmt += self.i_diag.split('TDU')[0][-1] + 'TDU'
                elif 'BRIDGE' in self.i_diag:
                    fmt += self.i_diag.split('BRIDGE')[0][-1] + 'BLP'
                filename = filename.replace('*',fmt+'_')
                if filename.endswith('_'):  # remove ending _
                    filename = filename[0:-1]
            if '{' in filename:
                filename = filename.format(**parm_dict)
            print('writing {fn}'.format(fn=filename))
            # Lukas had trouble with this in python 3
            self.write_DA(filename)
            
        if plot_DA and filename is not None:
            from pyfusion.data.DA_datamining import Masked_DA, DA
            self.da = DA(filename, load=True)
            # overplot if plot_DA > 1 on figure(plot_DA)
            axlist = plt.figure(plot_DA).gca() if plot_DA > 1 else None
            self.da.plot('Te', axlist=axlist, label_fmt='{{lab}}: Te {dt}'.format(dt=dtseg))

        if return_data:
            return(self.fitdata)

    # def process_constant_Langmuir removed 11/2018 -  not used, and confusing to
    # have two procedures so similar
            
# quick test code - just 'run' this file


if __name__ == '__main__':
    import sys
    # LP = Langmuir_data([20160310, 9], 'W7X_L57_LP01_04','W7X_L5UALL') # 4 chans
    # LP = Langmuir_data([20160310, 9], 'W7X_L53_LPALLI','W7X_L5UALL') # lower
    # LP = Langmuir_data([20160309, 7], 'W7X_L57_LPALLI','W7X_L5UALL') # upper
    # LP = Langmuir_data([20160302, 12], 'W7X_L57_LPALLI','W7X_L5UALL') # bad tb?
    # LP = Langmuir_data([20160310, 9], 'W7X_L57_LP01_02','W7X_L5UALL') #quickest
    # the following, using ([20160310, 9], 'W7X_L57_LP01_02) gives one nice char
    # and another char spoilt by a change in i_electron between cycles

    if len(sys.argv) < 2:
        # LP = Langmuir_data([20160309, 10], 'W7X_L57_LPALLI', 'W7X_L5UALL')
        LP = Langmuir_data([20160309, 10], 'W7X_L57_LP01_04', 'W7X_L5UALL')
        results = LP.process_swept_Langmuir(rest_swp=1, t_range=[0.91,0.92], t_comp=[0.05,0.1], threshchan=0, plot=3, return_data=True)
    elif sys.argv[1] == 'BRIDGE':
        print(' Very short file')
        LP = Langmuir_data([20180927,30], 'W7M_BRIDGE_ALLI','W7M_BRIDGE_V1', dev_name='W7M')
        results = LP.process_swept_Langmuir(t_comp=[0,0], t_range=[1.0,1.001],dtseg=2000, threshchan=-1,initial_TeVfI0=dict(Te=20,Vf=1,I0=None),fit_params=dict(cycavg=[200,10,-4]),return_data=True)

"""
Testing synchronisation using dead reckoning.
LP952.process_swept_Langmuir(overlap=1,dtseg=5e5/500.402/2,initial_TeVfI0=dict(Te=30,Vf=5,I0=None),fit_params=dict(alg='amoeba',maxits=300,lpf=21,esterr=1,track_ratio=1.2),filename='*2k2',threshold=0.001,threshchan=0)
# these overlap well about 20 samples difference beginning to end - could be the drift
plot(LP952.segs[-11][2].signal[1])
plot(LP952.segs[1][2].signal[1])

successful 9,7 session 797 IPP
run  pyfusion/examples\run_process_LP.py  replace_kw="dict(t_range=[0.0,0.7],t_comp=[-0.025,0],filename='c:/cygwin/tmp/*2k2am1p2_21')"  shot_list=[[20160309,7]] lpdiag='W7X_L5{s}_LPALLI' sweep_s
ig=W7X_L5UALL
796 IPP
run  pyfusion/examples\run_process_LP.py  replace_kw="dict(t_range=[0.0,0.7],t_comp=[-0.025,0],filename='c:/cygwin/tmp/*2k2am1p2_21')" seglist=['7'] shot_list=[[20160310,11]] lpdiag='W7X_L5{s}_LPALLI' sweep_sig=W7X_L57_U

"""
