"""
Pull the Langmuir probe characteristic data out of a graph for further 
plotting, processing

Testing:
# with the current figure containing a Langmuir characteristic and time plot,

run pyfusion/examples/get_LP_data.py "init=dict(Te=4,Vf=15,I0=None)" test=1

# Getting data from a plot produced by process_swept_Langmuir(plot=3_
run pyfusion/examples/get_LP_data.py
run pyfusion/examples/get_LP_data.py "init=dict(Te=4,Vf=15,I0=None)"

lpf    7      9     15     21    51   101     201
<res> 5e-4   3e-4  1e-4  6e-5   1e-5  3e-6    6e-7  
Te err  2%    1%
#_PYFUSION_TEST_@@Skip
"""
from __future__ import print_function
from six.moves import input

import sys
import matplotlib.pyplot as plt
import numpy as np
from pyfusion.data.process_swept_Langmuir import LPfitter

def LPchar( v, Te, Vf, I0):
    # hard to add a series resistance here as fun returns I as a fn of V
    return(I0 * (1 - np.exp((v-Vf)/Te)))
_var_defaults = """
lpf = 9  # approx harmonic at 3dB pt
test = 0
debug = 2  # >= 2 gives text info from leastsq (not implemented?)
ncycles = 2
lfnoise = [0,0,0,0]
noise = [.005, .5]
fit_params = {}
init = None
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if test:
    t = np.linspace(-.002,.002, 2000, endpoint=False)
    v = 50 * np.sin(2*np.pi*500*t)
    charparams = (40, 20, .02)
    i_probe = LPchar(v, *charparams)
    i_probe += noise[0] * 2*(np.random.random(len(t))-0.5)
    i_probe += (i_probe - max(i_probe)) * noise[1] * 2*(np.random.random(len(t))-0.5)
    label = str('test - {cp}'.format(cp=charparams))
    if np.max(np.abs(lfnoise))>0:
        N = len(i_probe)
        i_probe += np.sum([lfnoise[i] * np.sin(2*np.pi*i*np.arange(N)/float(N))
                           for i in range(len(lfnoise))],axis=0)

else:
    (labs_nums) = zip(plt.get_figlabels(),plt.get_fignums())
    for lab in labs_nums:
        label, num = lab
        if '[' in label:
            fig = plt.figure(label)
            mgr = plt.get_current_fig_manager()
            mgr.window.tkraise()
            break

    print('found "' + label + '"', end='')
    if label.startswith('['):
        print()
    else:
        ans = input(' which is not the expected figure label: continue?')
        if ans.lower() != 'y':
            sys.exit(1)

    i_probe, v = None, None

    ch = [c for c in fig.get_children() if hasattr(c,'get_axes') ]#and c.get_axes() is not None]
    if debug>2: print(ch)
    for cc in ch:
        chch = [c for c in cc.get_children() if hasattr(c, 'get_xydata') 
                and hasattr(c, 'get_label')]
        if debug>2: print(chch)
        for c in chch:
            if c.get_label() == 'i_probe':
                if i_probe is not None:
                    print('more than one iprobe found')
                t, i_probe = c.get_xydata().T

            if c.get_label() == 'v':
                if v is not None:
                    print('more than one v found')
                t, v = c.get_xydata().T
if i_probe is None or v is None:
    raise LookupError('missing data in ' + label)

LPcharax = plt.gca()

cnts,bins = np.histogram(i_probe, bins=20)
wneg = np.where(bins < 0)[0]
maxnegidx = np.argmax(cnts[wneg])
wnegsml = np.where((wneg > maxnegidx) & (cnts[wneg] < cnts[wneg[maxnegidx]]/3.))[0]
if debug:
    plt.figure()
    plt.hist(i_probe, bins=20)
    plt.plot(bins[wneg[wnegsml]], cnts[wneg[wnegsml]],'ro')
    plt.title('analysis of ' + label.replace('(','<'))
cutoff = bins[wneg[wnegsml]][0]
if debug:
    plt.show()  # block

input('CR to continue')

# figure()  # write on current fig.
fig, [axt, axfit] = plt.subplots(2, 1)
ft = np.fft.rfft(i_probe)

if lpf is None:
    f_filt = ft.copy() # to avoid corruption of the original when bads are deld
else:
    f_filt = ft*np.exp(-(np.arange(len(ft))/(1.7*ncycles*lpf))**2)


half =  chr(189).decode('latin1')
if ncycles == 2:
    badharm = range(1,40,2)
    goodharm = [2,4,6,8,10,12]
    frac = ['', half]
else:
    badharm = [] 
    goodharm = [1,2,3,4,5,6,7,8,9]
    frac = ['']
for n in badharm: f_filt[n]=0
i_filt = np.fft.irfft(f_filt)
axt.plot(i_filt)
fig.sca(axfit)  # back to the VI fit window
# this sets dummy.debug = (local value of debug)
class dummy:
    debug=debug

fitter = LPfitter(i_filt, v, parent=dummy, fit_params=fit_params)
[[Te, Vf, I0], resid, its, maxits] = fitter.fit(plot=2, init=init)

    
goodRMS = np.sqrt(np.sum(np.abs(ft)[goodharm]**2))
contam = np.sqrt(np.sum(np.abs(ft)[badharm]**2))/goodRMS
harms = unicode(',  '.join(['{n}{f}:{m:.0f}%'.decode('latin1')
                        .format(n=n//ncycles,f=frac[n % ncycles],
                                m=100*np.abs(ft[n]/goodRMS)) for n in badharm
                            if np.abs(ft[n]/goodRMS) > 0.1]))

extra = str('{h}th harm 3dB, resnrm={nr:.2g}, RMS contam={cpc:.0f}%'
            .format( h=lpf, cpc=contam*100, nr=resid/I0))
if len(harms)>10:
    extra += '\n'
extra += ' [' + harms + ']'
print(extra)
print("fitter.fit(init={'Vf': 15, 'I0': None, 'Te': 50}, plot=2, fit_params=dict(maxits=300))")
print("or for a new object, \nLPfitter(i_filt, v, parent=dummy).fit(init={'Vf': 15, 'I0': None, 'Te': 50}, plot=2, fit_params=dict(maxits=300))")

"""
Analyse trace - look for 50% increase in resid, and set errors to
half the max variation since then.  Works well for amoeba, but 
sometimes over estimated by leastsq approaching too quickly.

"""

self = fitter  # for convenience in writing code
trace = np.array(self.trace)
final_res = trace[-1][1]
wgt = np.where(trace[:,1] > 1.2* final_res)[0]
if len(wgt)<1:
    axt.set_title(extra)
    plt.show()
    raise ValueError('insufficient convergence to estimate errors')

ifrom = wgt[-1]
vals = np.array([elt[0].tolist() for elt in self.trace[ifrom:]])
est_errs = [(np.max(x)-np.min(x))/2. for x in vals.T]
errs = str(' esterrs {e}'.format(e=', ' .join(['{v:.2g}'.format(v=v) for v in est_errs])))
strip_h = 0.12
fig.subplots_adjust(bottom=.15 + strip_h, hspace=.35)
axcvg = fig.add_axes([0.145,0.05,0.78,strip_h])
axcvg.semilogy([t[1] for t in self.trace], label='resid')
axcvg.plot(ifrom, self.trace[ifrom][1],'o',label='{r:.1f} x err'
           .format(r=self.trace[ifrom][1]/self.trace[-1][1]))
axcvg.plot(0, 0,'-r',label='deltaTe')
axcvg2 = axcvg.twinx()
axcvg2.semilogy([abs(t[0][0]-Te) for t in self.trace],'r',label='deltaTe')
axcvg.legend(prop=dict(size='small'))
axt.set_title(extra + errs)
fig.suptitle(label)
plt.show()
plt.sca(LPcharax) # go back to window with orig VI curves











