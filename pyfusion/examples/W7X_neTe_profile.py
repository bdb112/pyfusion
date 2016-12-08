"""
exclude="['LP07']"   excludes L53_LP07 and L57_LP07
exclude="['L53_LP07','LP19']"   excludes L53_LP07 and both LP19s

Example:
run pyfusion/examples/W7X_neTe_profile.py dafile_list="['LP/all_LD/LP20160310_9_L5SEG_amoeba21_1.2_2k.npz']" exclude="['L57_LP{n:02d}'.format(n=n) for n in range(1,13)]"

run -i pyfusion/examples/W7X_neTe_profile.py dafile_list='["LP/all_LD/LP20160224_NN_L5SEG_2k2.npz".replace("NN",str(i)) for i in [30,38]]' labelpoints=0 t_range_list=[[0.1,0.2],[0.2,0.3]] diag2=ne18 av=np.median

t_range = [[x/1e3, (x+0.03)/1e3] for x in range(200,800,200)]


"""
import numpy as np
import matplotlib.pyplot as plt
import pyfusion
from pyfusion.data.DA_datamining import Masked_DA, DA
import sys
import os

from lukas.PickledAssistant import lookupPosition
from pyfusion.acquisition.W7X.mag_config import get_mag_config, plot_trim
from pyfusion.data.fits import fitgauss, mygauss, fitmirexp, mymirexp, fitbiexp, mybiexp

#  from cycler import cycler  # not working yet

#global debug, fake_sin, fake_ne, LRsym

debug = 0  # additional graphs showing all quantities used
fake_sin = 0
fake_ne = 0
TeLRsym = 0


if TeLRsym:
    dofit, fitfun = fitmirexp, mymirexp
else:
    dofit, fitfun = fitbiexp, mybiexp

from scipy.interpolate import griddata

def interpolate(xvals, yvals, ngrid=200):
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    
    #   take the bounds from the whole data set
    xgridded = np.mgrid[np.min(xvals): np.max(xvals): 1j*ngrid]
    # remove close pairs of points
    incinds = xvals.argsort()
    xvals, yvals = xvals[incinds], yvals[incinds]
    diffs = np.diff(xvals)
    # kind of works
    # farinds = np.where((np.abs(diffs) > 2) & ((np.abs(xvals[0:-1]) > 15) | (np.abs(xvals[0:-1]) <12)))[0]
    farinds = np.where((np.abs(diffs) > 2))[0]
    # this trick would miss the last one - so if the last found is one short, extend byone
    if max(farinds) == len(xvals) - 2:  # messy
        farinds = farinds.tolist()
        farinds.append(len(xvals)-1)
        farinds = np.array(farinds)
    print('*** selecting {f} from {all}'.format(f=len(farinds), all=len(xvals)))
    xvals, yvals = xvals[farinds], yvals[farinds]
    ygridded = griddata(xvals, yvals, xgridded, method='cubic')
    return(xgridded, ygridded)

def integrate(x, t):
    """
    >>> x = [1, 2, 2, 1]
    >>> t = [1, 2, 3, 4]
    >>> integrate(x, t)
    5.0
    """
    wgood = np.where(~np.isnan(x))[0]
    x = np.array(x)[wgood]
    t = np.array(t)[wgood]
    incinds = np.argsort(t)
    x = x[incinds]
    t = t[incinds]
    dt = np.diff(t)
    xmid = np.array((x[0:-1] + x[1:]))/2.0
    return(np.sum(xmid*dt))


def rot(x, y, t):
    from numpy import cos, sin
    return(x*cos(t)+y*sin(t), -x*sin(t)+y*cos(t))

def get_calculated(da, diag, inds, av=np.average, x=None, Tesmooth=None, nrm=None):
    import json
    ebars = None
    Vfkey = 'Vf' if 'Vf' in da.keys() else 'Vp'
    Vf = da[Vfkey]
    thispath = os.path.dirname(__file__)
    W7X_path = os.path.realpath(thispath+'/../acquisition/W7X/')
    fname = os.path.join(W7X_path,'limiter_geometry.json')
    limdata = json.load(open(fname))
    # Lukas' distances are in mm
    slp = np.polyval(np.polyder(limdata['shape_poly']), np.abs(x)/1000)
    sintheta = -np.sin(np.arctan(slp))
    # this is a smoother sintheta - accuracy drops outside of 50mm (too low)
    #  xx=linspace(-55,55,100)
    #  plot(xx, 6 + 55  - sqrt(55**2-(0.8*xx)**2),lw=3
    if fake_sin:
        print('fake_sin')
        sintheta = 0.01 * (6 + 55 - np.sqrt(55**2 - (0.8*np.array(x))**2))
    if diag == 'Pdens':
        # this forms the product first, then averages later - is this good?
        sig = da['I0'] * (Vf + 4 * da['Te']) * sintheta
        #if nrm is not None:
        #    sig = sig/nrm
        ebars = None
    elif diag == 'Pdsmooth':
        # using ne can avoid some noise at least on shot 30
        qe = 1.602e-19
        mp = 1.67e-27
        amu = da['info']['params']['amu']
        fact = 1/(0.6*qe)*np.sqrt(amu*mp/(qe))/1e18         # units of 1e18
        A = 1e-6
        # I0 = ne18 * A/fact * sqrt(te)
        # assume Te =Ti, vf is usually around 0.6 Te --> 4.5
        sig = da['ne18'] * A/fact * np.sqrt(Tesmooth) *  ( 4.6 *Tesmooth) * sintheta
        if fake_ne:
            sig = (5+0*da['ne18']) * np.power(Tesmooth, 1.5) * sintheta
            diag = diag + 'fake ne'
    else:
        raise LookupError('diagnostic {diag} not known'.format(diag=diag))
    if debug:
        plt.figure()
        plt.title(da.name + ' ' + diag)
        plt.plot(x, av(da['ne18'][inds],axis=0), 'o', label='ne18')
        plt.plot(x, av(sig[inds],axis=0), 'x', label='sig')
        plt.plot(x, Tesmooth, '+-', label='Te smooth')
        plt.plot(x, sintheta, '^', label='sintheta')
        if nrm is not None: 
            plt.plot(x, nrm, 'v', label='nrm')
            plt.plot(x, av(sig[inds]/nrm,axis=0), 'or', label='sig/nrm')
        xxx = np.linspace(min(x), max(x), 300)
        slpxxx = np.polyval(np.polyder(limdata['shape_poly']), np.abs(xxx)/1000)
        sinthetaxxx = -np.sin(np.arctan(slpxxx))
        plt.plot(x, 100* sintheta,'sg',label='100*sintheta')
        plt.plot(xxx, 100* sinthetaxxx)
        plt.legend()
        plt.show()
    excessive = 100 if debug>0 else 1000
    if np.max(av(sig[inds], axis=0)) > excessive*np.median(sig[inds]):
        raise ValueError('Suspiciously high value ({v:.1g}) of {diag}'
                         .format(diag=diag, v=np.max(av(sig[inds], axis=0))))
    return(sig, ebars)


def LCFS_plot(da, diag, t0_utc, t_range, av=None, ax=None, xtoLCFS=1, nrm=None, labelpoints=False, col=None, Tesmooth=None):
    """ plot Te ne etc vs distance to LCFS, averaging over a time range.
    The average uses a Nanmean by default, so it does not de-emphasize 
    values with large uncertainties.
    Alternatively np.median can be used - but it treats nans as large => bias up
    """
    av = np.nanmean if av is None else av
    ax = plt.gca() if ax is None else ax
    chans = np.array(da['info']['channels'])
    dt = (t0_utc - da['info']['params']['i_diag_utc'][0])/1e9
    indrange = np.searchsorted(da['t_mid'], np.array(t_range) + dt)
    inds = range(*indrange)
    if len(inds) < 3:
        ll = len(inds)
        pyfusion.logging.warn('{msg} samples ({ll}) in the time range {t}'
                              .format(msg=['!No! ','too few'][ll>0], ll=ll, t=t_range))
    distLCFS = []
    td = []         # td is transverse signed distance from limiter vertical midplane 

    if diag in  ['ne18','Pdens','I0']:
        exclude.extend(['LP11','LP12','L57_LP09'])
    for (c, ch) in enumerate(chans):
        LPnum = int(ch[-2:])
        lim = 'lower' if 'L57' in ch else 'upper'
        X, Y, Z, dLCFS = lookupPosition(LPnum, lim)
        td.append(rot(X, Y,  2*np.pi*4/5.)[1])
        distLCFS.append(dLCFS)

    if diag in ['ne18', 'Te', 'I0', 'Vf', 'Vp']:
        if diag=='Vf' and 'Vp' in da.keys():
            diag = 'Vp'
        sig = av(da.masked[diag][inds], axis=0)
        if 'e'+diag in da:
            ebars = av(da['e'+diag][inds], axis=0)
        else:
            ebars = None
    else:
        sig, ebars = get_calculated(da, diag, inds, av=av, x=td, nrm=nrm, Tesmooth=Tesmooth)
        sig = av(sig[inds], axis=0)

        #print()
    # normalise before plots - intended for ne and I0 to allow for probe wear
    if nrm is not None:
        sig = sig/nrm
        if debug>0:
            print(nrm)

    signed_dLCFS = distLCFS * np.sign(td)
    for (c, ch) in enumerate(chans):
        if sum([st in ch for st in exclude]):
            sig[c] = np.nan
        if labelpoints>0 and (not np.isnan(sig[c])):
            if labelpoints>1 or sig[c]>1.3*np.median(sig):
                ax.text(signed_dLCFS[c], sig[c], ch)
    ax.errorbar([td, signed_dLCFS][xtoLCFS], sig, ebars, fmt='o', label=lim, color=col, lw=0.5)
    ax.set_ylabel(diag)
    #if diag == 'Te':
    ax.set_ylim(0, ax.get_ylim()[1])
    return(signed_dLCFS, td, sig, ebars)

def infostamp(txt, fig=None, default_kwargs=dict(horizontalalignment='right', verticalalignment='bottom', fontsize=7, x=0.99, y=0.008), **kwargs):
    import os, sys, pyfusion

    fig = plt.gcf() if fig is None else fig
    actual_kwargs = default_kwargs.copy()
    actual_kwargs.update(kwargs)
    extra = ' '.join(['pyfusion V ', pyfusion.VERSION,os.getlogin(),os.getcwd(), sys.version[:12]])
    plt.figtext(s=txt+' '+extra, figure=fig, **actual_kwargs)

#global debug, fake_sin, fake_ne, LRsym

_var_defaults = """
dev_name = 'W7X'
dafile_list = ['LP/all_LD/LP20160309_52_L5SEG_2k2.npz']
labelpoints=0
diag1='Te'
diag2='ne18'
exclude = []
use_t_mid=0 # if true, time is relative to t_mid
av = np.median
nrms = [None, None]
fits = 'Te'
save_images = 'None'
Te_lim = None
ne_lim = None
fp_list = [sys.stdout, open('profile_log.log','ab')]
xtoLCFS = 1
axset_list = "None"
#fake_ne = 0
#TeLRsym = 1
#debug=0
#fake_sin

#da53 = DA('LP/LP20160309_10_L53_2k2.npz')
#da57 = DA('LP/LP20160309_10_L57_2k2.npz')
t_range = [0.91, 0.93]
t_range = [0.51, 0.53]
# t_range = [[x/1e3, (x+0.03)/1e3] for x in range(200,800,200)]

# t_range = [0.61, 0.63]
# t_range = [1.15, 1.18]
#da53 = DA('LP/LP20160309_52_L53_2k2.npz')
#da57 = DA('LP/LP20160309_52_L57_2k2.npz')
t_range_list = [0.5, 0.52]
#t_range = [0.45, 0.47]
#t_range = [0.48, 0.5]
#t_range = [0.3, 0.32]
"""

exec(_var_defaults)

# read single arg if there are no equal signs - otherwise interpret options
if len(sys.argv)>1 and np.all(['=' not in sa for sa in sys.argv]):
    dafile_list=sys.argv[1]
else:
    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())


def get_half_width(params):
    hw = []
    for i in [0,1]:
        if params[4+i]<0: # if the offset is negative, the fit is probably a striaght line
            halfW = (params[2+i]+params[4+i])/(params[i]*params[2+i])/2.
        else:
            halfW = np.log(2)/params[i]
        hw.append(halfW)
    return(hw)

#    dafile = 'LP/all_LD/LP20160310_7_L5SEG_2k2.npz'
if not isinstance(dafile_list, (list, tuple)):
    dafile_list = [dafile_list]

if not isinstance(t_range_list[0], (list, tuple)):
    t_range_list = len(dafile_list) * [t_range_list]

if len(t_range_list)>1 and len(dafile_list) == 1:
    dafile_list = len(t_range_list) * dafile_list

# figLCFS, (axLCTe, axLCne) = plt.subplots(2, 1, sharex='all')
ns = len(dafile_list)
if axset_list is 'None':
    axset_list = ns * [None]
elif not isinstance(axset_list, (list, tuple)):
    if axset_list == 'row':
        nr = ns; nc = 2
    elif axset_list == 'column':
        nr = 2; nc = ns
    figLCFS, axs = plt.subplots(nrows=nr, ncols=nc, sharex='all', squeeze=None)
    axset_list = [[figLCFS, ax] for ax in axs]

else:
    print('assuming we have been given a valid list in unput')


print(dafile_list)
for framenum, (dafile, t_range, axset) in enumerate(zip(dafile_list, t_range_list, axset_list)):
    da53,da57 = [DA(dafile.replace('SEG',n)) for n in ['3','7']]


    shot_number = [da53['date'][0], da53['progId'][0]]
    dev = pyfusion.getDevice(dev_name)
    if not use_t_mid:
        echdata = dev.acq.getdata(shot_number,'W7X_TotECH')
        wech = np.where(echdata.signal > 100)[0]
        tech = echdata.timebase[wech[0]]
        t0_utc = int(tech * 1e9) + echdata.utc[0]
    else:
        t0_utc = da57['info']['params']['i_diag_utc'][0]

    if axset is None:
        figLCFS, (axLCTe, axLCne) = plt.subplots(2, 1, sharex='all')
    else:
        figLCFS,  (axLCTe, axLCne) = axset

    for ax in (axLCTe, axLCne):
        ax.set_color_cycle(['b', 'r', 'y', 'g', 'orange', 'c', 'm'])
        locator = MaxNLocator(prune='upper', nbins=4) # trim nbins to keep numbers on x axis round
        ax.yaxis.set_major_locator(locator)


    xlabel = ['Horizontal distance from limiter centre (mm)','Distance into LCFS (mm, from lukas R. -ve is left side)'][xtoLCFS]
    kwargs = dict(t0_utc=t0_utc, t_range=t_range, labelpoints=labelpoints, av=av, xtoLCFS=xtoLCFS)

    solds = []
    tds = []
    sigs = []
    fitsigs = []
    maxTe = 0
    maxne = 0

    for seg, col, da, nrm in zip(['3','7'],['b','r'], [da53, da57], nrms):
        for fp in fp_list:
            fp.write('\n{fn}: {d}'.format(fn=da.name, d=diag2))
        kwargs.update(dict(col=col))
        sold, td, sig1, ebar1 = LCFS_plot(da, diag1, ax=axLCTe, **kwargs)
        sigs.append(sig1)
        tds.append(td)
        solds.append(sold)
        if diag1 in fits: 
            params, cond, fitsig1 = dofit(sold, sig1, yerrs=ebar1, ax=None)
            xgridded, ygridded = interpolate([td, sold][xtoLCFS], sold)

            ycurve = fitfun(ygridded, params)
            if xtoLCFS == 0:
                omit = np.where(np.abs(xgridded) < 14)[0]
                xgridded[omit] = np.nan

            axLCTe.plot(xgridded, ycurve, color=col, lw=1.5)
            if len(params) == 3:
                params6 = [params[0], params[0], params[1], params[1], params[2], params[2]]
            else:
                params6 = params

            TehalfL, TehalfR = get_half_width(params6)         
            for fp in fp_list:
                fp.write('\n=={sh}=====> Seg {s} Te0 = {Te0:.1f}, Te18 = {Te18:.1f}, TeWid = {TeWL:.1f} {TeWR:.1f}, ofs L,R={ofsL:.1f} {ofsR:.1f}  t_range={tr}'
                  .format(s=seg, Te18=np.average(fitfun(np.array([-18,18]), params)),
                          Te0=np.average(fitfun(np.array([-1,1]), params)),
                          TeWL=TehalfL, TeWR=TehalfR, ofsL=params6[4], ofsR=params6[5],
                          tr=t_range, sh=shot_number))
            fitsigs.append(fitsig1)

        sold, td, sig2, ebar2 = LCFS_plot(da, diag2, nrm=nrm, ax=axLCne, Tesmooth=fitsig1, **kwargs)
        sigs.append(sig2)
        tds.append(td)
        solds.append(sold)
        for fp in fp_list:
            fp.write(', Integral =  {intg:.1f}\n'.format(intg=integrate(sig2, td)))

        # if diag1 in fits: params = fitgauss(sold, sig1, ebar1, ax=axLCne)

        maxTe = max(maxTe, np.nanmax(sig1))
        maxne = max(maxne, np.nanmax(sig2))
        #1/0
    cfg, cfg_dict = get_mag_config(shot_number)
    if cfg_dict is not None:
        plot_trim(axLCTe, mag_dict=cfg_dict, aspect=2.6, size=0.06, color='g')
        ind = 13 - 12*cfg[2]/.3904
        if abs(ind - round(ind)) < .03:
            ind = int(round((ind)))
        ind = 'Ind {ind}'.format(ind=ind)
    else:
        ind = ''

    tb = 'LP t_mid' if use_t_mid else 'into ECH'
    figLCFS.subplots_adjust(hspace=0, right=0.87)
    # figLCFS.suptitle('shot {s}, {avname} over time {fr}-{t}s {tb}: {ind}' 
    title = str('shot {s}, {avname} filter over time {fr}-{t}s {tb}: {ind}' 
                .format(s=[da['date'][0], da['progId'][0]],
                        fr=t_range[0],t=t_range[1], tb=tb,
                        avname=av.__name__, ind=ind))
    if axset is None:
        axLCTe.set_title(title)
        leg_title=''
    else:
        leg_title=title
    infostamp(' '.join([da53.name, da57.name]))

    for maxval, ax in zip([maxTe, maxne],[axLCTe, axLCne]):
        if ax != axLCne:
            ax.legend(prop=dict(size='medium'),#title=leg_title, 
                loc='upper right')
            #        loc=['best','lower left'][save_images is not 'None'])
            # lower left best for ne, upper right for Te
            #ax.set_xlabel('distance to LCFS (from Lukas R: -ve for left side)')
        ax.text(0.5, 0.03, leg_title, transform=ax.transAxes,
               fontsize='small', horizontalalignment='center')
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, maxval*1.03)
        if ax == axLCTe and Te_lim is not None:
            ax.set_ylim(Te_lim)

        if ax == axLCne and ne_lim is not None:
            ax.set_ylim(ne_lim)

        #ax.set_prop_cycle(cycler('color', ['b', 'r', 'y', 'k']))

    figLCFS.show()
    if save_images is not 'None':
        figLCFS.savefig(save_images+'{fr:03}.png'.format(fr=framenum))
        if framenum>10: figLCFS.clear()  # don't take up too many - keep 10

fp_list[-1].close()
""" 
figure(); i3=argsort(solds[0]); plot(solds[0][i3], sigs[1][i3]*sqrt(fitsigs[0][i3]),'o-'); i7=argsort(solds[2]); plot(solds[2][i7], sigs[3][i7]*sqrt(fitsigs[1][i7]),'o-');ylim(0,ylim()[1]); title(da.name+' '+str(t_range))
"""
