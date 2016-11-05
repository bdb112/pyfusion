""" Complete processing of swept Langmuir data, using two classes.
New version allows choice of algorithm and adds more params
This supersedes the example in examples/process_swept_Langmuir.py
 
pyfusion v0.7.0 - estimate error, include a lpfilter, resid normed to I0
This version assumes that all required sweep data are in a multi-channel diagnostic (dataset)
 - the other version (not in the repo) grabs sweepV data as required

    Example:
    run pyfusion/data/process_swept_Langmuir
    LP.process_swept_Langmuir()
    LP.write_DA('20160310_9_L57')
    # then to tune mask:
    from pyfusion.data.DA_datamining import Masked_DA, DA
    myDA=DA('20160310_9_L57',load=1)  # just to be sure it is a good DA
    myDA.masked=Masked_DA(['Te','I0','Vp'], baseDA=myDA)
    myDA.da['mask']=(myDA['resid']/abs(myDA['I0'])<.35) & (myDA['nits']<100)
    clf();plot(myDA.masked['Te']);ylim(0,100)


"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin
import os
import pyfusion
from pyfusion.utils.time_utils import utc_ns
import pyfusion.conf as conf
from scipy.fftpack import hilbert
from pyfusion.data import amoeba
from scipy.optimize import leastsq
import time
from pyfusion.debug_ import debug_
from pyfusion.utils.primefactors import nice_FFT_size
from pyfusion.conf.utils import get_config_as_dict

debug = 0


def AC(x):
    ns = len(x)
    return(x - np.average(x * np.blackman(ns))/np.average(np.blackman(ns)))


def fixup(da=None, locs=None, newmask=None, suppress_ne=None, newpath='/tmp'):
    """ some miscellaneous post processing mainly to set the time offset of ECH
    need to run plot_LP2d.py first.
    sets t_zero, and tweaks the Nan coverage
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
def LPchar(v, Te, Vp, I0):
    # hard to add a series resistance here as fun returns I as a fn of V
    return(I0 * (1 - exp((v-Vp)/Te)))

def residuals(params, y, x):
    Te, Vp, I0 = params
    err = LPchar(x, Te, Vp, I0) - y
    # return np.sqrt(np.abs(err))
    return err
"""

class LPfitter():
    def __init__(self, i, v,  fit_params=None, plot=1, verbose=1, parent=None, debug=1):
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

    def LPchar(self, v, Te, Vp, I0):
        # hard to add a series resistance here as fun returns I as a fn of V
        return(I0 * (1 - exp((v-Vp)/Te)))

    def plotchar(self, v, Te, Vp, I0, alpha=1, col='g', linewidth=2):
        varr = np.linspace(np.min(v), np.max(v))
        plt.plot(varr, self.LPchar(varr, Te, Vp, I0), 'k', linewidth=linewidth+1)
        plt.plot(varr, self.LPchar(varr, Te, Vp, I0), col, linewidth=linewidth)

    def error_fun(self, vars, data):
        """ return RMS error """
        Te, Vp, I0 = vars
        v = data['v']
        i = data['i']
        Lnorm = self.actual_fparams['Lnorm']
        varr = np.linspace(np.min(v), np.max(v))
        if self.plot > 3:
            # note - alpha chosen according to number of lines (maxits)
            #  We use the maxits value here via actual_fparams
            # we would prefer to use the actual number used
            plt.plot(varr, self.LPchar(varr, Te, Vp, I0), 'g', linewidth=2,
                     alpha=min(1, 1./np.sqrt(self.actual_fparams['maxits'])))
        #  err = np.sqrt(np.mean((i - LPchar(v, Te, Vp, I0))**2))
        err = np.power(np.mean(np.abs(i - self.LPchar(v, Te, Vp, I0))**Lnorm), 1./Lnorm)
        if self.debug > 4:
            print('error fun:'+', '.join(['{k} = {v:.3g}'.format(k=var, v=val)
                             for (var, val) in dict(Te=Te, Vp=Vp, I0=I0, resid=err).iteritems()]))
        self.trace.append([np.copy(vars), err])
        return(-err)  # amoeba is a maximiser, hence - sign


    def residuals(self, params, x, y):
        Te, Vp, I0 = params
        err = self.LPchar(x, Te, Vp, I0) - y
        Lnorm = self.fit_params['Lnorm']
        self.trace.append([params.copy(), np.power(np.mean(np.abs(err)**Lnorm), 1./Lnorm)])
        if  Lnorm != 2:  # leastsq routine will square the residual
            err = np.power(np.abs(err), Lnorm/2.)
        return err
        
    def fit(self, init=None, plot=None, fit_params=None, default_init=dict(Te=50, Vp=15, I0=None)):
        """ Perform a curve fit operation with the given parameter and initial
        value.  'None' invokes the fit_params determined at creation
        of the class, and for 'init', the default_init

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
        Vp = init['Vp']
        I0 = init['I0']
        if I0 is None:
            I0 = 1.7 * np.average(np.clip(self.i, 0, 1000))

        var = [Te, Vp, I0]
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
                    print(ier, msg)

        elif alg == 'amoeba':
            fit_results = list(amoeba.amoeba(var, scale, self.error_fun, 
                                             itmax=maxits, ftolerance=ftol*I0,
                                             xtolerance=xtol,
                                             data=dict(v=self.v, i=self.i)))
            fit_results[1] = -fit_results[1]  # change residual back to positive
            s_sq = None  # don't know how to get cov for amoeba
            pcov = None

        Te, Vp, I0 = fit_results[0]
        fit_results[1] /= I0  # normalise to IO 
        residual, mits = fit_results[1:3]
        fit_results.append(maxits)
        self.pcov = pcov
        self.s_sq = s_sq

        if plot > 1:
            plt.scatter(self.v, self.i, c='r', s=80,
                        alpha=min(1, 4/np.sqrt(len(self.v))))

            self.plotchar(self.v, Te, Vp, I0, linewidth=4)
            # provide brief info about algorithm and Lnorm in the title
            plt.title('Te = {Te:.1f}eV, Vp = {Vp:.1f}, Isat = {I0:.3g}, <res> = {r:.2e} {mits} {A}{p}:its '
                      .format(Te=Te, Vp=Vp, I0=I0, r=residual, mits=mits, A=alg.upper()[0],p=Lnorm))
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
    tryone([Te,Vp,I0],dict(v=sweepV,i=iprobe))
    """
    # tryone([25,-15, .02],dict(i=iprobe, v=sweepV))
    print(-error_fun(var, data))
    Te, Vp, I0 = var
    v = data['v']
    i = data['i']
    plotchar(v, Te, Vp, I0, linewidth=w)


def find_clipped(sigs, clipfact):
    """ look for digitizer or amplifier saturation in all raw
    signals, and return offending indices.  Digitizer saturation
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
    if a dictionary params is supplied, use these to process the data, 
    otherwise just read in the data.

    Eternal question: Should I pass parameters or set them in the object?
    Answer for now - if you want them in the record of analysis, pass 
    them through process_swept_Langmuir

    obvious candidates for object: debug, plot, select
    obvious candidates for args: Lnorm, initial_TeVpI0, clipfact, rest_swp

    If params contains a filename entry, save the result as that name

    Final solution: save the ACTUAL params used in the object, and then 
    save THOSE in when writing, not the onbes entered.
    """
    def __init__(self, shot, i_diag, v_diag, dev_name="W7X", debug=debug, plot=1, verbose=0, params=None):
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

        self.imeasfull = self.dev.acq.getdata(shot, i_diag)
        self.vmeasfull = self.dev.acq.getdata(shot, v_diag)
        comlen = min(len(self.vmeasfull.timebase), len(self.imeasfull.timebase))
        FFT_size = nice_FFT_size(comlen-2, -1)
        # the minus 2 is a fudge to hide small inconsistencies in reduce_time
        # e.g. 20160310 9 W7X_L5_LPALLI
        self.imeasfull = self.imeasfull.reduce_time([self.imeasfull.timebase[0], self.imeasfull.timebase[FFT_size]])
        self.vmeasfull = self.vmeasfull.reduce_time([self.vmeasfull.timebase[0], self.vmeasfull.timebase[FFT_size]])

        if self.params is not None:
            self.process_swept_Langmuir(**self.params)


    def get_iprobe(self, leakage=None, t_comp=None):
        """ main purpose is to subtract leakage currents
        Will use the full data, as the t_range is meant to be plasma interval
        returns a copy of the measured courremt, overwritten with the
        corrected iprobe
        """
        # obtain leakage estimate
        if t_comp is None:
            t_comp = self.t_comp
        FFT_size = nice_FFT_size(len(self.imeasfull.timebase), -1)
        self.iprobefull = self.imeasfull.copy()
        self.sweepQ = []  # will keep these for synchronous sampling
        input_leakage = leakage
        for (c, chan) in enumerate(self.imeasfull.channels):
            if self.select is not None and c not in self.select:
                continue
            leakage = input_leakage
            cname = chan.config_name
            sweepV = self.vcorrfull.signal[self.vlookup[self.vassoc[c]]][0:FFT_size]
            sweepQ = hilbert(sweepV)
            self.sweepQ.append(sweepQ)  # save for synchronising segments (it is smoothed)

            # these attempts to make it accept a single channel are only partial
            imeas = self.imeasfull.signal[c] # len(self.imeasfull.channels) >1 else self.imeasfull.signal
            tb = self.imeasfull.timebase

            w_comp = np.where((tb>=t_comp[0]) & (tb<=t_comp[1]))[0]
            if len(w_comp) < 2000:

                raise ValueError('Not enough points {wc} t_comp - try {tt}'
                    .format(tt=np.round([tb[0], tb[0] + t_comp[1]-t_comp[0]],3),
                            wc=len(w_comp)))
            ns = len(w_comp)
            wind = np.blackman(ns)
            offset = np.mean(wind * imeas[w_comp])/np.mean(wind)
            sweepVFT = np.fft.fft(AC(sweepV[w_comp]) * wind)
            imeasFT = np.fft.fft(AC(imeas[w_comp]) * wind)
            ipk = np.argmax(np.abs(sweepVFT)[0:ns//2])  # avoid the upper one
            comp = imeasFT[ipk]/sweepVFT[ipk]

            #print('leakage compensation factor = {r:.2e} + j{i:.2e}'
            #      .format(r=np.real(comp), i=np.imag(comp)))
            print('{u}sing computed leakage comp factor = {m:.2e} e^{p:.2f}j'
                  .format(u = ["Not u", "U"][leakage is None],
                          m=np.abs(comp), p=np.angle(comp)))
            if leakage is None:
                leakage = [np.real(comp), np.imag(comp)]

            # find the common length - assuming they start at the same time????
            comlen = min(len(self.imeasfull.timebase),len(self.vmeasfull.timebase),len(sweepQ))
            # put signals back into rdata (original was copied by reduce_time)
            # overwrite - is this OK?
            self.iprobefull.signal[c] = self.iprobefull.signal[c]*0.  # clear it
            # sweepV has a DC component! beware
            self.iprobefull.signal[c][0:comlen] = self.imeasfull.signal[c][0:comlen]-offset \
                                        - sweepV[0:comlen] * leakage[0] - sweepQ[0:comlen] * leakage[1]
            # remove DC cpt (including that from the compensation sweepV)
            offset = np.mean(wind * self.iprobefull.signal[c][w_comp])/np.mean(wind)
            self.iprobefull.signal[c][0:comlen] -= offset

    def prepare_sweeps(self, rest_swp='auto', sweep_freq=500, Vpp=90*2, clip_level_minus=-88):
        """ extracts sweep voltage data for all probes,
        call restore_sin if necessary
        Initially, the voltage data is assumed to be in a dictionary
        """
        if self.debug > 0:
            print('entering prepare_sweeps ', len(self.vmeasfull.signal[0]))
        if str(rest_swp).lower() == 'auto':
            rest_swp = self.shot[0]>20160309
            print ('* Automatically setting rest_swp to {r} *'.format(r=rest_swp))

        if rest_swp:
            from pyfusion.data.restore_sin import restore_sin
        self.vcorrfull = self.vmeasfull.copy()
        if rest_swp:
            for (c, sig) in enumerate(self.vmeasfull.signal):
                self.vcorrfull.signal[c] = restore_sin(self.vmeasfull, chan=c, sweep_freq=sweep_freq,
                                                       Vpp=Vpp, clip_level_minus=clip_level_minus, verbose=self.verbose)

    def fit_swept_Langmuir_seg_multi(self, m_seg, i_seg, v_seg, clipfact=5,  initial_TeVpI0=None, fit_params=None, plot=None):
        if self.select is None:
            self.select = range(len(self.i_chans))
        res = []
        for (c, chan) in enumerate(i_seg.channels):
            if self.select is not None and c not in self.select:
                continue
            v = v_seg.signal[self.vlookup[self.vassoc[c]]]
            i = i_seg.signal[c]
            im = m_seg.signal[c]
            result = self.fit_swept_Langmuir_seg_chan(im, i, v, i_seg, channame=chan.config_name, clipfact=clipfact,  initial_TeVpI0=initial_TeVpI0, fit_params=fit_params, plot=plot)
            if fit_params.get('esterr', False):             
                [Te, ne, I0], resid, nits, maxits, Ie_Ii, errest = result
                eTe, ene, eI0 = errest
            else:
                [Te, ne, I0], resid, nits, maxits, Ie_Ii = result

            t_mid = np.average(i_seg.timebase)
            res_list = [t_mid, Te, ne, I0, resid, nits, maxits, Ie_Ii]
            if fit_params.get('esterr', False):             
                res_list.extend( [eTe, ene, eI0])
            res.append(res_list)
        return(res)
    
    def fit_swept_Langmuir_seg_chan(self, m_sig, i_sig, v_sig, i_segds, channame, fit_params=None, clipfact=5,  initial_TeVpI0=None, plot=None):
        """Lowish level routine - takes care of i clipping and but not
        leakage or restoring sweep.

        Returns fitted params including error
    """
        # Note that we check the RAW current for clipping
        try:  # this relative import doesn't work when process is run (for test)
            from pyfusion.data.LPextra import lpfilter, estimate_error
        except ValueError:
            from pyfusion.data.LPextra import lpfilter, estimate_error

        lpf = fit_params['lpf'] if fit_params is not None and 'lpf' in fit_params else None
            
        segtb = i_segds.timebase
        v, i, im = v_sig.copy(), i_sig.copy(), m_sig.copy()     # for fit
        # v_n, i_n, im_n = v * np.nan, i * np.nan, im * np.nan,   # for nan plots

        good = find_clipped([v, im], clipfact=clipfact)
        wg = np.where(good)[0]
        if (~good).any():  # if any bad
            if len(wg) < 0.3 * len(v) or len(wg) < 32:  # was (~good).all():
                print('{t:.3f}s suppressing all!*** {a}'
                      .format(a=len(good),t=np.average(segtb)))
                if fit_params.get('esterr', False):
                    return([3*[None], None, None, None, None, 3*[None]]) # incl dummy errsa
                else:
                    return([3*[None], None, None, None, None])

            elif self.verbose > 0 or len(wg) < 0.9 * len(v):
                print('{t:.3f}s suppress clipped {b}/{a}'
                      .format(b=len(np.where(~good)[0]), a=len(good), t=np.average(segtb)))

        # [[Te, Vp, I0], res, its]
        if plot is None:
            # plot in detail only if there are 20 figures or less
            plot = self.plot + 2*((len(list(self.segs))*len(self.imeasfull.channels)) < 8)
        if (plot >= 2) and len(self.figs) < 8:  # 20 really slows down
            # best to open in another session.
            interval = str('{t}'.format(t=[round(segtb[0], 6), round(segtb[-1], 6)]))
            fig, axs = plt.subplots(nrows=1 + (plot >= 3), ncols=1, squeeze=0,\
                                    num='{i} {c}{s}'.format(i=interval, c=channame, s=self.actual_params['suffix']))

            self.figs.append(fig)
            if plot >= 3:
                # these labels help find the data
                axs[1, 0].plot(segtb, i, 'b', lw=.2, label='i_probe')
                axs[1, 0].plot(segtb[wg], i[wg], 'b.')
                axv = axs[1, 0].twinx()
                axv.set_ylabel('sweep V')
                axv.plot(segtb, v, 'r', lw=.2, label='v')
                axv.plot(segtb[wg], v[wg], 'r.')
            plt.sca(axs[1, 0])  # ready to be overlaid by filtered I

            fig.suptitle('{shot} {tr} {d}:{c}'
                         .format(tr=interval, shot=self.shot, d=self.i_diag, c=channame))

        #res = LPfit(v, i, plot=(debug > 1),maxits=100)
        if lpf is not None:
            i = lpfilter(segtb, i, v, lpf=lpf, debug=self.debug, plot=self.plot)
            # check if i has been truncated to an even number by real fft - 
            if wg[-1] >= len(i):  # if so, truncate the other pieces
                wg = wg[0:-1]
                v = v[0:-1]
                segtb = segtb[0:-1]
        if (plot >= 2) and len(self.figs) < 8:
            plt.sca(axs[0, 0])  # ready for VI curve

        self.fitter = LPfitter(i=i[wg], v=v[wg], fit_params=fit_params, plot=plot, parent = self)
        result = self.fitter.fit(init=initial_TeVpI0, fit_params=fit_params, plot=plot)
        # get a 'fitted' value of most negative current to give us Ie_max
        # Use the fitted curve evaluated at the max voltage (measured V is smooth)
        maxv = np.max(v)
        debug_(self.debug, 2, key='after_fit')
        Te, Vp, I0 = result[0]
        Imaxsmoothed = self.fitter.LPchar(maxv, Te, Vp, I0)
        Ie_Ii = (I0-Imaxsmoothed)/I0
        result.append(Ie_Ii)
        #  print(Te, result)
        if fit_params.get('esterr', False):
            if fit_params['esterr'] != 'cov':
                self.pcov = None  # signal routine not to use covariance    
            result.append(estimate_error(self, result, debug=self.debug))
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
        lookup = [(0, 't_mid'), (1, 'Te'), (2, 'Vp'), (3, 'I0'), 
                  (4, 'resid'), (5, 'nits'), (6, 'maxits'), (7, 'Ie_Ii'),
                  (3, 'ne18')]

        if self.fitter.fit_params.get('esterr',False):
            lookup.extend([(8, 'eTe'), (9, 'eVp'), (10, 'eI0') ])

        for (ind, key) in lookup:
            if key not in dd:
                dd[key] = np.zeros([nt, nc], dtype=np.float32)
            dd[key][:] = res[:, :, ind]

        # fudge t_mid is not a vector...should fix properly
        dd['t_mid'] = dd['t_mid'][:, 0]
        dd['info'] = dict(params=self.actual_params,
                          coords=[self.coords[ic] for ic in self.select],
                          shotdata=dict(shot=[self.shot], utc_ns=[self.imeas.utc[0]]),
                          channels=[chn.replace(self.dev.name+'_', '')
                                    .replace('_I', '')
                                    for chn in
                                    [self.i_chans[ic] for ic in self.select]],
                          orig_name = os.path.split(filename)[-1],
                          username = os.getlogin())
        
        da = DA(dd)
        da.masked = Masked_DA(['Te', 'I0', 'Vp', 'ne18', 'Ie_Ii'], baseDA=da)
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
                         & (np.abs(da['Vp']) < 200) & (np.abs(da['Te']) < 200) 
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

    def process_swept_Langmuir(self, t_range=None, t_comp=[0, 0.1], fit_params = dict(maxits=200, alg='leastsq',esterr=1), initial_TeVpI0=dict(Te=50, Vp=15, I0=None), dtseg=4e-3, overlap=1, rest_swp='auto', clipfact=5, leakage=None, threshold=0.01, threshchan=12, filename=None, amu=1, plot=None, return_data=False, suffix=''):
        """ 
        ==> results[time,probe,quantity]
        plot = 1   : V-I and data if there are not too many.
        plot >= 2  : V-I curves
        plot >= 3  : ditto + time plot
        plot >= 4  : ditto + all I-V iterations

        can send parameters through the init or process - either way
         they all are recorded in actual_params

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
        # first do the things that are better done on the whole data set.

        self.figs = []  # reset the count of figures used to stop too many plots
        self.actual_params = locals().copy()
        for k in fit_params:
            if k not in 'Lnorm,cov,esterr,alg,xtol,ftol,lpf,maxits,track_ratio'.split(','):
                raise ValueError('Unknown fit_params key ' + k)
        self.actual_params.pop('self')
        self.actual_params.update(dict(i_diag=self.i_diag, v_diag=self.v_diag))
        for k in self.actual_params:
            if k not in 'amu,clipfact,dtseg,filename,initial_TeVpI0,leakage,overlap,plot,rest_swp,suffix,t_comp,t_range,threshold,threshchan,fit_params,v_diag,i_diag,return_data'.split(','):
                raise ValueError('Unknown actual_params key ' + k)

        self.actual_params.update(dict(i_diag_utc=self.imeasfull.utc, pyfusion_version=pyfusion.VERSION))
        self.amu = amu
        if not isinstance(self.imeasfull.channels, (list, tuple, np.ndarray)):
            self.imeasfull.channels = [self.imeasfull.channels]

        self.i_chans = [ch.config_name for ch in self.imeasfull.channels]
        for ch in self.i_chans:  # only take the current channels, not U
            if ch[-1] != 'I':
                raise ValueError("Warning - removal of V chans doesn't work!!!")
                # hopefully we can ignore _U channels eventually, but not yet
                self.i_chans.remove(ch)

        self.coords = [ch.coords.w7_x_koord for ch in self.imeasfull.channels]
        # if you want just one voltage channel, at the moment, you still
        # need it to be a multi-channel diag. (to simplify this code).
        self.v_chans = [ch.config_name for ch in self.vmeasfull.channels]

        # want to say self.vmeas[ch].signal where ch is the imeas channel name
        self.vlookup = {}
        for (c, vch) in  enumerate(self.v_chans):
            self.vlookup[vch] = c

        self.vassoc = []  # list of sweepVs associated with each i channel
                          # one per i channel - these refer to their 
                          # respective self.v_chans
                          # for OP1.1, only a few V chans were recorded and 
                          # in practice only two channels are necessary.  
        default_sweep = 'NO SWEEP'
        default_sweep = 'W7X_L57_LP01_U'

        for ch in self.i_chans:
            cd = get_config_as_dict('Diagnostic', ch)
            # TODO(bdb): use of default_sweep should generate a warning
            self.vassoc.append(cd.get('sweepv', default_sweep))

        # prepare_sweeps will populate self.vcorrfull
        self.prepare_sweeps(rest_swp=rest_swp)
        self.get_iprobe(leakage=leakage, t_comp=t_comp)

        tb = self.iprobefull.timebase
        # the 3000 below tries to avoid glitches from Hilbert at both ends
        #w_plasma = np.where((np.abs(self.iprobefull.signal[threshchan]) > threshold) & (tb > tb[3000]) &(tb < tb[-3000]))[0]
        # only look at electron current - bigger (shot 0309.52 LP53 has a positive spike at 2s)
        w_plasma = np.where((-self.iprobefull.signal[threshchan] > threshold) & (tb > tb[3000]) &(tb < tb[-3000]))[0]
        
        if t_range is None:
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

        if self.debug>0: print(' {n} segments'.format(n=len(self.segs)))
        self.fitdata = []
        debug_(self.debug, 3, key='process_loop')
        for mseg, iseg, vseg in self.segs:
            # print('len vseg', len(vseg.signal[0]))
            if len(vseg.signal[0]) < dtseg//5:  # skip short segments
                continue
            print(np.mean(mseg.timebase),4,end='s: ')
            self.fitdata.append(self.fit_swept_Langmuir_seg_multi(mseg, iseg, vseg, clipfact=clipfact, initial_TeVpI0=initial_TeVpI0, fit_params=fit_params, plot=plot))
        # note: fitter.actual_fparams only records the most recent!
        self.actual_params.pop('fit_params') # only want the ACTUAL ones.
        self.actual_params['actual_fit_params'] = self.fitter.actual_fparams
        if filename is not None:            
            if  '*' in filename:
                fmt = 'LP{s0}_{s1}_'
                if 'L5' in self.i_diag:
                    fmt += 'L5' + self.i_diag.split('L5')[1][0]
                filename = filename.replace('*',fmt+'_')
                if filename.endswith('_'):  # remove ending _
                    filename = filename[0:-1]
            if '{' in filename:
                filename = filename.format(s0=self.shot[0], s1=self.shot[1], i_diag=self.i_diag)
            print('writing {fn}'.format(fn=filename))
            self.write_DA(filename)
            
        if return_data:
            return(self.fitdata)

# quick test code - just 'run' this file
if __name__ == '__main__':
    #LP = Langmuir_data([20160310, 9], 'W7X_L57_LP01_04','W7X_L5UALL') # 4 chans
    #LP = Langmuir_data([20160310, 9], 'W7X_L53_LPALLI','W7X_L5UALL') # lower
    #LP = Langmuir_data([20160309, 7], 'W7X_L57_LPALLI','W7X_L5UALL') # upper
    #LP = Langmuir_data([20160302, 12], 'W7X_L57_LPALLI','W7X_L5UALL') # bad tb?
    LP = Langmuir_data([20160310, 9], 'W7X_L57_LP01_02','W7X_L5UALL') #quickest
    # the following, using ([20160310, 9], 'W7X_L57_LP01_02) gives one nice char
    # and another char spoilt by a change in i_electron between cycles
    results = LP.process_swept_Langmuir(rest_swp=1,t_range=[1.6,1.604],t_comp=[0.8,0.85],threshchan=0)

"""
Testing synchronisation using dead reckoning.
LP952.process_swept_Langmuir(overlap=1,dtseg=5e5/500.402/2,initial_TeVpI0=dict(Te=30,Vp=5,I0=None),fit_params=dict(alg='amoeba',maxits=300,lpf=21,esterr=1,track_ratio=1.2),filename='*2k2',threshold=0.001,threshchan=0)
# these overlap well about 20 samples difference beginning to end - could be the drift
plot(LP952.segs[-11][2].signal[1])
plot(LP952.segs[1][2].signal[1])
"""
