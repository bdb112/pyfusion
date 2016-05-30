""" Complete processing of swept Langmuir data, using two classes.
New version allows choice of algorithm - version after this will allow more params
This supersedes the example in examples/process_swept_Langmuir.py
 
This version assumes that all required sweep data are in a multi-channel diagnostic (dataset)
 - the other version (not in the repo) grabs sweepV data as required

    Example:
    run pyfusion/data/process_swept_Langmuir
    LP.process_swept_Langmuir()
    LP.write_DA('20160310_9_L57')
    # then to tune mask:
    from pyfusion.data.DA_datamining import Masked_DA, DA
    da=DA('20160310_9_L57',load=1)  # just to be sure it is a good DA
    da.masked=Masked_DA(['Te','I0','Vp'], DA=da)
    da.da['mask']=(da['resid']/abs(da['I0'])<.35) & (da['nits']<100)
    clf();plot(da.masked['Te']);ylim(0,100)


"""
import pyfusion
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin
import pyfusion.conf as conf
from scipy.fftpack import hilbert
from pyfusion.data import amoeba
from scipy.optimize import leastsq
import time
from pyfusion.debug_ import debug_
from pyfusion.utils.primefactors import nice_FFT_size
from pyfusion.conf.utils import get_config_as_dict

debug = 1


def AC(x):
    ns = len(x)
    return(x - np.average(x * np.blackman(ns))/np.average(np.blackman(ns)))

# Nasty! in order to make leastsq work we need these OUT of a class
def LPchar(v, Te, Vp, I0):
    # hard to add a series resistance here as fun returns I as a fn of V
    return(I0 * (1 - exp((v-Vp)/Te)))

def residuals(params, y, x):
    Te, Vp, I0 = params
    err = LPchar(x, Te, Vp, I0) - y
    # return np.sqrt(np.abs(err))
    return err

class LPfitter():
    def __init__(self, i, v, pnorm=1, maxits=None, plot=1, verbose=1, parent=None, debug=1):
        self.i = i
        self.v = v
        self.debug = debug
        self.verbose = verbose
        self.plot = plot
        self.pnorm = pnorm
        self.maxits = 100 if maxits is None else maxits
        self.parent = parent
        self.trace = []

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
        varr = np.linspace(np.min(v), np.max(v))
        if self.plot > 3:
            plt.plot(varr, self.LPchar(varr, Te, Vp, I0), 'g', linewidth=2,
                     alpha=min(1, 1./np.sqrt(self.maxits)))
        #  err = np.sqrt(np.mean((i - LPchar(v, Te, Vp, I0))**2))
        err = np.power(np.mean(np.abs(i - self.LPchar(v, Te, Vp, I0))**self.pnorm), 1./self.pnorm)
        if self.debug > 4:
            print(', '.join(['{k} = {v:.3g}'.format(k=var, v=val)
                             for (var, val) in dict(Te=Te, Vp=Vp, I0=I0, resid=err).iteritems()]))
        self.trace.append([vars, -err])
        return(-err)  # amoeba is a maximiser

    """
    def residuals(params, y, x):
        Te, Vp, I0 = params
        err = self.LPchar(x, Te, Vp, I0) - y
        # return np.sqrt(np.abs(err))
        return err
    """       
    def fit(self, init=dict(Te=50, Vp=15, I0=None), plot=None, maxits=None, alg='leastsq'):
        # Takes about 2ms/it for 2500 points, and halving this only saves 10%
        # if maxits is None:  # this trickiness makes it work, but not pythonic.
        #    maxits = globals()['maxits']
        self.trace = []
        if plot is None:
            plot = self.plot

        if maxits is None:
            maxits = self.maxits

        Te = init['Te']
        Vp = init['Vp']
        I0 = init['I0']
        if I0 is None:
            I0 = 1.7 * np.average(np.clip(self.i, 0, 1000))

        var = [Te, Vp, I0]
        scale = np.array([Te, Te/3, I0])
        if alg == 'leastsq':
            fit_results, cov, infodict, msg, ier = \
                leastsq(residuals, var, args=(self.i, self.v), maxfev=maxits, full_output=1)
            resid = np.power(np.mean(np.abs(infodict['fvec']**self.pnorm)),1/self.pnorm)
            fit_results = [fit_results, resid, infodict['nfev']]
        elif alg == 'amoeba':
            fit_results = list(amoeba.amoeba(var, scale, self.error_fun, 
                                             itmax=maxits, ftolerance=I0*1e-4,
                                             xtolerance=1e-4,
                                             data=dict(v=self.v, i=self.i)))
            fit_results[1] = -fit_results[1]  # change residual back to positive

        Te, Vp, I0 = fit_results[0]
        residual, mits = fit_results[1:3]
        fit_results.append(maxits)
        if plot > 1:
            plt.scatter(self.v, self.i, c='r', s=80,
                        alpha=min(1, 4/np.sqrt(len(self.v))))

            self.plotchar(self.v, Te, Vp, I0, linewidth=4)
            plt.title('Te = {Te:.1f}eV, Vp = {Vp:.1f}, Isat = {I0:.3g}, resid = {r:.2e} {mits} its '
                      .format(Te=Te, Vp=Vp, I0=I0, r=residual, mits=mits))
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
    Ansewr for now - if you want them in the record of analysis, passthem through process_swept_Langmuir

    obvious candidates for object: debug, plot, select
    obvious candidates for args: pnorm, initial_TeVpI0, clipfact, rest_swp

    If params contains a filename entry, save the result as that name
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
        input_leakage = leakage
        for (c, chan) in enumerate(self.imeasfull.channels):
            if self.select is not None and c not in self.select:
                continue
            leakage = input_leakage
            cname = chan.config_name
            sweepV = self.vcorrfull.signal[self.vlookup[self.vassoc[c]]][0:FFT_size]
            sweepQ = hilbert(sweepV)

            # these attempts to make it accept a single channel are only partial
            imeas = self.imeasfull.signal[c] # len(self.imeasfull.channels) >1 else self.imeasfull.signal
            tb = self.imeasfull.timebase

            w_comp = np.where((tb>=t_comp[0]) & (tb<=t_comp[1]))[0]
            ns = len(w_comp)
            sweepVFT = np.fft.fft(AC(sweepV[w_comp]) * np.blackman(ns))
            imeasFT = np.fft.fft(AC(imeas[w_comp]) * np.blackman(ns))
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
            self.iprobefull.signal[c][0:comlen] = self.imeasfull.signal[c][0:comlen] \
                                        - sweepV[0:comlen] * leakage[0] - sweepQ[0:comlen] * leakage[1]

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

    def fit_swept_Langmuir_seg_multi(self, m_seg, i_seg, v_seg, clipfact=5,  initial_TeVpI0=None, maxits=None, plot=None):
        res = []
        for (c, chan) in enumerate(i_seg.channels):
            if self.select is not None and c not in self.select:
                continue
            v = v_seg.signal[self.vlookup[self.vassoc[c]]]
            i = i_seg.signal[c]
            im = m_seg.signal[c]
            [Te, ne, I0], resid, nits, maxits = self.fit_swept_Langmuir_seg_chan(im, i, v, i_seg, channame=chan.config_name, clipfact=clipfact,  initial_TeVpI0=initial_TeVpI0, maxits=maxits, plot=plot)
            t_mid = np.average(i_seg.timebase)
            res.append([t_mid, Te, ne, I0, resid, nits, maxits])
        return(res)
    
    def fit_swept_Langmuir_seg_chan(self, m_sig, i_sig, v_sig, i_segds, channame, maxits=None, clipfact=5,  initial_TeVpI0=None, plot=None):
        """Lowish level routine - takes care of i clipping and but not
        leakage or restoring sweep.

        Returns fitted params including error
    """
        # Note that we check the RAW current for clipping
        segtb = i_segds.timebase
        v, i, im = v_sig.copy(), i_sig.copy(), m_sig.copy()     # for fit
        # v_n, i_n, im_n = v * np.nan, i * np.nan, im * np.nan,   # for nan plots

        good = find_clipped([v, im], clipfact=clipfact)
        wg = np.where(good)[0]
        if (~good).any():  # if any bad
            if len(wg) < 0.3 * len(v):  # was (~good).all():
                print('{t:.3f}s suppressing all!*** {a}'
                      .format(a=len(good),t=np.average(segtb)))
                return([3*[None], None, None, None])
            elif self.verbose > 0 or len(wg) < 0.9 * len(v):
                print('{t:.3f}s suppress clipped {b}/{a}'
                      .format(b=len(np.where(~good)[0]), a=len(good), t=np.average(segtb)))

        # [[Te, Vp, I0], res, its]
        if plot is None:
            # plot in detail only if there are 20 figures or less
            plot = self.plot + 2*((len(self.segs)*len(self.imeasfull.channels)) < 8)
        if (plot >= 2) and len(self.figs) < 10:  # 20 really slows down
            # best to open in another session.
            interval = str('{t}'.format(t=[round(segtb[0], 6), round(segtb[-1], 6)]))
            fig, axs = plt.subplots(nrows=1 + (plot >= 3), ncols=1, squeeze=0,\
                                    num='{i} {c}'.format(i=interval, c=channame))
            self.figs.append(fig)
            if plot >= 3:
                # these labels help find the data
                axs[1, 0].plot(segtb, i, 'b', lw=.2, label='i_probe')
                axs[1, 0].plot(segtb[wg], i[wg], 'b.')
                axv = axs[1, 0].twinx()
                axv.set_ylabel('sweep V')
                axv.plot(segtb, v, 'r', lw=.2, label='v')
                axv.plot(segtb[wg], v[wg], 'r.')
            plt.sca(axs[0, 0])  # ready for VI curve

            fig.suptitle('{shot} {tr} {d}:{c}'
                         .format(tr=interval, shot=self.shot, d=self.i_diag, c=channame))

        #res = LPfit(v, i, plot=(debug > 1),maxits=100)
        self.fitter = LPfitter(i=i[wg], v=v[wg], maxits=maxits, plot=plot, parent = self)
        return(self.fitter.fit(init=initial_TeVpI0, plot=plot))

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
        for (ind, key) in [(0, 't_mid'), (1, 'Te'), (2, 'Vp'), (3, 'I0'), 
                           (4, 'resid'), (5, 'nits'), (6, 'maxits'), (3, 'ne18')]:
            if key not in dd:
                dd[key] = np.zeros([nt, nc], dtype=np.float32)
            dd[key][:] = res[:, :, ind]

        # fudge t_mid is not a vector...should fix properly
        dd['t_mid'] = dd['t_mid'][:, 0]
        dd['info'] = dict(params=self.actual_params,
                          coords=self.coords,
                          shotdata=dict(shot=[self.shot], utc_ns=[self.imeas.utc[0]]),
                          channels=[chn.replace(self.dev.name, '')
                                    .replace('_I', '')
                                    for chn in self.i_chans])
        da = DA(dd)
        da.masked = Masked_DA(['Te', 'I0', 'Vp', 'ne18'], DA=da)
        #  da.da['mask']=(da['resid']/abs(da['I0']) < .7) & (da['nits']<100)
        da.da['mask'] = ((da['resid']/abs(da['I0']) < .7) & (da['nits'] < da['maxits'])
                         & (np.abs(da['Vp']) < 200) & (np.abs(da['Te']) < 200))
        qe = 1.602e-19
        mp = 1.67e-27
        fact = 1/(0.6*qe)*np.sqrt(self.amu*mp/(qe))/1e18         # units of 1e18
        # check if each channel has an area
        for (c, chn) in enumerate(self.i_chans):
            cd = get_config_as_dict('Diagnostic', chn)
            A = float(cd.get('area', 1.8e-6))
            da.da['ne18'][:, c] = fact/A * da['I0'][:, c]/np.sqrt(da['Te'][:, c])
        da.save(filename)

    def process_swept_Langmuir(self, t_range=None, t_comp=[0, 0.1], maxits=200, initial_TeVpI0=dict(Te=50, Vp=15, I0=None), dtseg=4e-3, overlap=1, rest_swp='auto', clipfact=5, leakage=None, threshold=0.01, filename=None, amu=1, plot=None):
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
        self.actual_params.pop('self')
        self.actual_params.update(dict(i_diag=self.i_diag, v_diag=self.v_diag))
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

        self.v_chans = [ch.config_name for ch in self.vmeasfull.channels]
        # want to say self.vmeas[ch].signal where ch is the imeas channel name
        self.vlookup = {}
        for (c, vch) in  enumerate(self.v_chans):
            self.vlookup[vch] = c

        self.vassoc = []  # list of associated sweepVs - one per i channel
        default_sweep = 'NO SWEEP'
        default_sweep = 'W7X_L57_LP01_U'

        for ch in self.i_chans:
            cd = get_config_as_dict('Diagnostic', ch)
            self.vassoc.append(cd.get('sweepv', default_sweep))

        # prepare_sweeps will populate self.vcorrfull
        self.prepare_sweeps(rest_swp=rest_swp)
        self.get_iprobe(leakage=leakage, t_comp=t_comp)

        tb = self.iprobefull.timebase
        # the 3000 below tries to avoid glitches from Hilbert at both ends
        w_plasma = np.where((np.abs(self.iprobefull.signal[0]) > threshold) & (tb > tb[3000]) &(tb < tb[-3000]))[0]
        
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
            self.fitdata.append(self.fit_swept_Langmuir_seg_multi(mseg, iseg, vseg, clipfact=clipfact, initial_TeVpI0=initial_TeVpI0, maxits=maxits, plot=plot))
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
    results = LP.process_swept_Langmuir(rest_swp=1,t_range=[1.6,1.604],t_comp=[0.8,0.85])
