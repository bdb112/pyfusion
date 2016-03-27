""" process sinusoidally swept Langmuir probes (W7X) 

debug=1 normal

# good compensation
# No! - wrong gains run pyfusion/examples/process_swept_Langmiur.py diag_name=W7X_L57_LP7_I "atten=(1.015e-4,-2e-5)"

Best so far 10 Ohm!!
run pyfusion/examples/process_swept_Langmuir.py diag_name=W7X_L57_LP7_I "atten=(3.4e-4,-6.5e-5)" maxits=1000 t_range=[1,1.1] debug=3

Best so far! 1Ohm!!
run pyfusion/examples/process_swept_Langmuir.py diag_name=W7X_L57_LP10_I "atten=(.5e-5,-5e-5)" maxits=1000 t_range=[1,1.1] debug=3

Play around after
LPfit(sweepV[inds], iprobe[inds],plot=1,maxits=20, init=dict(Te=3,Vp=0,I0=None))
"""
# n_e = I/(q*A) * sqrt(2*pi*mi/(q*Te))    # Chen notes
# n_e = I/(0.6*q*A)*sqrt(mi/(q*Te))/1e18  #  usual -> 2.4e19 for 100mA
import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, sin
import pyfusion.conf as conf
from scipy.fftpack import hilbert
from pyfusion.data import amoeba
import time
from pyfusion.debug_ import debug_

def AC(x):
    ns = len(x)
    return(x - np.average(x * np.blackman(ns))/np.average(np.blackman(ns)))

def LPchar(v, Te, Vp, I0):
    # hard to add a series resistance here as fun returns I as a fn of V
    return(I0 * (1 - exp((v-Vp)/Te)))

def plotchar(v, Te, Vp, I0, alpha=1, col='g', linewidth=2):
    varr = np.linspace(np.min(v), np.max(v))
    if debug > 0 + 2*(len(t_range) == 1) :   # if debug>3 will do all segs
        plt.plot(varr,  LPchar(varr, Te, Vp, I0), 'k', linewidth=linewidth+1)
        plt.plot(varr,  LPchar(varr, Te, Vp, I0), col, linewidth=linewidth)


def error_fun(var, data):
    """ return RMS error """
    Te, Vp, I0 = var
    v = data['v']
    i = data['i']
    varr = np.linspace(np.min(v), np.max(v))
    if debug > 1: 
        plt.plot(varr,  LPchar(varr, Te, Vp, I0), 'g', linewidth=2, 
                 alpha=min(1,1./np.sqrt(maxits)))
    #err = np.sqrt(np.mean((i - LPchar(v, Te, Vp, I0))**2))
    err = np.power(np.mean(np.abs(i - LPchar(v, Te, Vp, I0))**pnorm),1./pnorm)
    if debug>2: 
        print(', '.join(['{k} = {v:.3g}'.format(k=i[0],v=i[1]) 
                         for i in dict(Te=Te, Vp=Vp, I0=I0, resid=err).iteritems()]))
    return(-err)  #  amoeba is a maximiser

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

# tryone([25,-15, .02],dict(i=iprobe, v=sweepV))
def tryone(var, data, mrk='r', w=4):
    """ evaluate fit and show curve - meant for manual use
    tryone([Te,Vp,I0],dict(v=sweepV,i=iprobe))
    """
    print(-error_fun(var, data))
    Te, Vp, I0 = var
    v = data['v']
    i = data['i']
    plotchar(v, Te, Vp, I0, linewidth=w)

def find_clipped(sigs, fact):
    """ look for digitizer or amplifier saturation in all raw 
    signals, and return offending indices.  Digitizer saturation
    is easy, amplifier saturation is softer - need to be careful 

    sigs can be a signal, array or a list of those
    Note that restore_sin() can be used on the sweep, rather than omitting
    the points.
    """
    if isinstance(sigs,(list, tuple)):
        goodbits = np.ones(len(sigs[0]),dtype=np.bool)
        for sig in sigs:
            goodbits = goodbits & find_clipped(sig, fact)
            debug_(debug, 5, key='find_clipped')

        return(goodbits)
    
    cnts, vals = np.histogram(sigs, 50)   # about 100ms for 250k points
    
    if (np.sum(cnts[0]) > fact * np.sum(cnts[1])):
        wunder = (sigs < vals[1])
    else:
        wunder = np.zeros(len(sigs),dtype=np.bool)

    if (np.sum(cnts[-1]) > fact * np.sum(cnts[-2])):
        wover =  (sigs > vals[-2])
    else:
        wover = np.zeros(len(sigs),dtype=np.bool)

    return(~wunder & ~wover)


def LPfit(v,i,init=dict(Te=50, Vp=15, I0=None), plot=None, maxits=None):
    #Takes about 2ms/it for 2500 points, and halving this only saves 10%
    if maxits is None:  # this trickiness makes it work, but not pythonic.
        maxits = globals()['maxits']
    if plot is None: plot = debug>0

    Te = init['Te']
    Vp = init['Vp']
    I0 = init['I0']
    if I0 is None:
        I0 = 1.7 * np.average(np.clip(i, 0, 1000))

    var = [Te, Vp, I0]
    scale = np.array([Te, Te/3, I0])
    fit_results = amoeba.amoeba(var, scale, error_fun, itmax = maxits, 
                                data = dict(v=v, i=i))
    Te, Vp, I0 = fit_results[0]
    plotchar(v, Te, Vp, I0, linewidth=4)
    residual, mits = fit_results[1:3]
    if plot: 
        plt.title('Te = {Te:.1f}eV, Vp = {Vp:.1f}, Isat = {I0:.3g}, resid = {r:.2e} {mits} its '
                       .format(Te=Te, Vp=Vp, I0=I0, r=-residual, mits=mits))
        plt.suptitle('{shot} {tr} {d}'
                     .format(tr=t_range, shot=shot_number, d=diag_name))
        plt.show(block=0)

    return(fit_results)


_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_L57_LP3_I"
shot_number = [20160308,25]
sharey=2
plotkws={}
atten = None  # None is automatic
fun=myiden
dphi = 0.0  # shift between measured current and voltage - only for testing
fun2=myiden2
hold=0
debug=1
t_range=[1.34,1.36] # if a list of a single time, it is a dt. If three, they are args to arange
t_subrange=None
t_comp = [0.1,0.2] #  range for subtraction of leakage signal
threshold = 0.001  # current in Amps above which the plasma is assumed to be
pnorm=1  #  pnorm = 2 is like RMS, pnorm is like avg(abs()), seems best     
maxits=100
sweep_default="W7X_L57_LP1_U"
restore_sweep = -88  # if not 0, try to restore clipped sin for values below this
fact = 5
"""
locs_before = locals().copy()
exec(_var_defaults)
locs_after = locals().copy()

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

input_params = {}
for var in locs_after:
    if not var in locs_before:
        input_params.update({var: locals()[var]})

for k in 'locs_before'.split(','):
    input_params.pop(k)

input_params['version'] = pyfusion.version.get_version()


if not isinstance(t_range, (tuple, list, np.ndarray)):
    t_range = [t_range]  # make it a list for convenience in tests

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name)

if debug > 1 and hold == 0: 
    plt.figure()
if debug > 1: data.plot_signals(suptitle='shot {shot}: '+diag_name, sharey=sharey,
                  fun=fun, fun2=fun2, **plotkws)
imeas = data.signal
params = conf.utils.get_config_as_dict('Diagnostic', diag_name)
if 'sweep' in params:
    sweep_diag = params['sweep']
else:
    if debug>0:
        sweep_diag = sweep_default
        print('defaulting sweep_diag to {dsd}'.format(dsd=sweep_diag))

sweep_data =  dev.acq.getdata(shot_number, sweep_diag)
dt = sweep_data.utc[0] - data.utc[0]
if dt != 0:
    print('******* Warning - timebases differ by {dt:.3f} us'
          .format(dt=dt/1e3))

# replace by time overlap, using reduce time.
# always reduce time to make sure a copy is made.
# this reduction is to select the common area of the signals
sweepVmeas = sweep_data.signal
dats = [sweep_data, data]
common_start = max([x.timebase[0] for x in dats])
common_end = min([x.timebase[-1] for x in dats])
# the first reduce time
sweepVmeas_rdata = sweep_data.reduce_time([common_start, common_end])
imeas_rdata = data.reduce_time([common_start, common_end])
tb = imeas_rdata.timebase
print('timebase length = {l:,d}'.format(l=len(tb)))

# first need to shift IMeas relative to sweep if there is an offset.  
# Can use hilbert to shift the sweep instead.
sweepQ = hilbert(sweepVmeas_rdata.signal)
sweepV = sweepVmeas_rdata.signal
if dphi != 0:
    print('Warning - changing sweep phase this way corrupts the DC cpt!!!')
    sweepV = cos(dphi) * AC(sweepVmeas_rdata.signal) - sin(dphi) * sweepQ

imeas = imeas_rdata.signal

w_comp = np.where((tb>=t_comp[0]) & (tb<=t_comp[1]))[0]
ns = len(w_comp)
sweepVFT = np.fft.fft(AC(sweepV[w_comp]) * np.blackman(ns))
imeasFT = np.fft.fft(AC(imeas[w_comp]) * np.blackman(ns))
ipk = np.argmax(np.abs(sweepVFT)[0:ns//2])  # avoid the upper one
comp = imeasFT[ipk]/sweepVFT[ipk]

print('leakage compensation factor = {comp}'.format(comp=comp))
if atten is None:
    atten = [np.real(comp), np.imag(comp)]

if debug>1:
    plt.plot(tb, sweepV*atten[0],'r',label='sweepV')
    mainax = plt.gca()
    plt.plot(tb, imeas - sweepV*atten[0],'c', label='corrected for R')

# put signals back into rdata (original was copied by reduce_time)
# overwrite - is this OK?
imeas_rdata.signal = imeas
from copy import deepcopy
iprobe_rdata = deepcopy(imeas_rdata)
iprobe = imeas_rdata.signal - sweepV*atten[0] - sweepQ * atten[1]
iprobe_rdata.signal = iprobe
tb = imeas_rdata.timebase
# Hilbert transform creates a transient at 0 - skip first and last 3000
w_plasma = np.where((np.abs(iprobe) > threshold) & (tb > tb[3000]) & (tb < tb[-3000]))[0]
if t_subrange is None:
    t_subrange = [tb[w_plasma[0]], tb[w_plasma[-1]]]

if debug>1:
    plt.plot(tb, iprobe,'g',label='R and phase')
    plt.xlim(t_subrange[0]-.05, t_subrange[0]+.05)
    if debug > 1: mainax.plot(t_subrange, [0, 0],'c*-', linewidth=3)
    plt.legend(prop=dict(size='small'))

if restore_sweep:
    from pyfusion.data.restore_sin import restore_sin
    debug_(debug, 4, key='before_restore_sweep')
    # for now, don't reduce time in restore_sin, because we want both single and multi segments to work
    sweepV = restore_sin(sweepVmeas_rdata, clip_level_minus=restore_sweep, t_range=None)
    # for now, put it back in the data object - a little messy - is it legal?
    # probably better to return a new data object, as it could be
    # faster to reduce the time base to a nicer number for FFTs
    sweepVmeas_rdata.signal = sweepV

if len(t_range) == 2:  # ==> one manual fit - not an auto series
    inds = np.where((tb>t_range[0]) & (tb<t_range[1]))[0] 

    sweepV = sweepV[inds]
    tb = tb[inds]
    imeas = imeas[inds]
    iprobe = iprobe[inds]

    # doesn't matter if we look for clipping in restore vsweep - there shouldn't be any if it is restored
    good = find_clipped([sweepV, imeas],fact=fact)
    if (~good).any():
        print('suppressing {b}/{a}'
              .format(b=len(np.where(~good)[0]), a=len(good)))
    sweepV = sweepV[np.where(good)[0]]
    iprobe = iprobe[np.where(good)[0]]
    tb = tb[np.where(good)[0]]

    if debug>0: 
        plt.figure()
        plt.scatter(sweepV,iprobe,c='r', s=80,
                 alpha=min(1,4/np.sqrt(len(inds))))

    ylims = plt.ylim()
    results = LPfit(sweepV,iprobe)
    plt.ylim(ylims)

    if debug>0:
        plt.figure()
        plt.plot(tb, sweepV*np.max(np.abs(atten)))
        plt.plot(tb, iprobe)
        plt.show(block=0)

else:  # series of automated fits
    ipdr = iprobe_rdata.reduce_time(t_subrange)
    imdr = imeas_rdata.reduce_time(t_subrange)
    sdr = sweepVmeas_rdata.reduce_time(t_subrange)

    segs = zip(sdr.segment(t_range[0]),
               ipdr.segment(t_range[0]),
               imdr.segment(t_range[0]))
    print(' {n} segments'.format(n=len(segs)))
    tresults = []
    for vseg, iseg, mseg in segs:
        stb = vseg.timebase
        v = vseg.signal
        i = iseg.signal
        # Note that we check the RAW current for clipping
        good = find_clipped([v,mseg.signal],fact=fact)
        if (~good).any():
            print('suppressing {b}/{a}'
                  .format(b=len(np.where(~good)[0]), a=len(good)))
            v = v[np.where(good)[0]]
            i = i[np.where(good)[0]]
            tb = stb[np.where(good)[0]]

        if debug > 3:
            plt.figure(num=str('{t}'.format(t=[vseg.timebase[0],vseg.timebase[-1]])))
            plt.scatter(v,i,c='r', s=80,
                        alpha=min(1,4/np.sqrt(len(v))))

        [[Te, Vp, I0], res, its] =  LPfit(v, i, plot=(debug>1))
        tresults.append([(vseg.timebase[0] + vseg.timebase[1])/2.,
                         Te, Vp, I0, res, its])

    import pickle
    tail = ['']
    for i in range(1,50):
        tail.append(str('_{i}'.format(i=i)))

    for t in tail:
        fn = str('LP_{shot[0]}_{shot[1]}_{diag_name}_{nsamp}_{today}{t}.pickle'
                 .format(shot=shot_number, diag_name=diag_name, 
                     today = time.strftime('%Y%m%d',time.gmtime()),
                         nsamp = len(segs[1][1].timebase), t=t))
        if not os.path.isfile(fn):
            break

    print('saving in '+ fn)
    pickle.dump(dict(tresults=tresults, params=input_params), open(fn,'wb'))

plt.show(block=0)

