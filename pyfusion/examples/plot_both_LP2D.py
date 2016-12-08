""" Script to animate Te and ne for both segments - see plot_LP2D.py for one 
To make larger, nicer fonts 
    import poster_setup   

fig.savefig('EILT_bdb_200.jpg',format='jpeg',dpi=200)
fig.savefig('EILT_bdb_300.jpg',format='jpeg',dpi=300)
fig.savefig('EILT_bdb_300.png',dpi=300)
fig.savefig('EILT_bdb_200.png',dpi=200)
fig.savefig('EILT_bdb_200.pdf',dpi=200)

axtime.set_xlim(-.15,.75)
axtwin.set_ylim(0,60)
fig.savefig('EILT_bdbR_200.pdf',dpi=200)


"""
#_PYFUSION_TEST_@@Skip
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
from pyfusion.data.DA_datamining import DA
import pyfusion
from pyfusion.acquisition.W7X.puff_db import puff_db
from matplotlib.ticker import MaxNLocator
from pyfusion.acquisition.W7X.puff_db import get_puff
from pyfusion.acquisition.W7X.mag_config import get_mag_config, plot_trim
from copy import deepcopy

"""
dafile = '20160302_12_L57'
Te_range = None
ne_range = None

dafile = '20160302_12_L57'
Te_range = [10, 100]  # 20160302_12
ne_range = [0, 2]

dafile = 'LP20160302_12_L57'
Te_range = [10, 100]  # 20160302_12
ne_range = [0, 2]


"""
db = {}

db.update(dict(LP20160310_9=dict(
Te_range = [20, 100],  # 
ne_range = [0, 15],
    p_range = [0,5],  # 3 later in shot
minpts=15,
srange = range(294,320), # .35 sec
#srange = range(600,620), # 983ms
#srange = range(680,700) # 1140ms
)))


db.update(dict(LP20160310_7=dict(
Te_range = [10, 80],  # 
ne_range = [0, 2],
minpts=16,
#srange = range(200,220), #440
srange = range(400,420), #840
#srange = range(800,820), # 1.644 sec 
)))
"""
dafile = 'LP20160310_39_L57'
Te_range = [10, 100]  # 
ne_range = [0, 4]


dafile = 'LP20160308_41_L57' # also 44
Te_range = [10, 100]  # 
ne_range = [0, 6]

dafile = 'LP20160309_42_L5SEG'
Te_range = [10, 60]  # 53 [10,70]
ne_range = [0, 10]   # 53 [0,10]
srange = range(60, 62)   # 20160309_42_L53
minpts=18

dafile = 'LP20160218_4_L57_2k2'
Te_range = [5, 15]  # 53 [10,70]
ne_range = [0, 20]   # 53 [0,10]
srange = range(60, 72)   # very low Te
srange = range(40, 52)   #  lowish Te  puff?
probe_chans = [0,1]
minpts=18


dafile = 'LP20160309_10_L53_amoeba21_1.2_2k.npz'
dafile = 'LP20160309_10_L57_2k2.npz'
Te_range = [10, 80]  
ne_range = [0, 3.5]   
srange = range(60, 92)   # early High power region
#srange = range(200, 232)   # mid lower power
# srange = range(430, 442)   # 920ms
#t_range = [0.92, 0.93]
minpts=18
probe_chans = [1,6]

dafile = 'LP20160309_52_L53_2k2.npz'
#dafile = 'LP20160309_52_L57_2k2.npz'
dafile = 'LP20160309_51_L53_2k2.npz'
dafile = 'LP20160309_51_L57_2k2.npz'
"""
db.update(dict(LP20160309_52=dict(
Te_range = [5, 20],  # [8, 60] for 51
ne_range = [0, 15],
#probe_chans = [1,4,5,6,7],
    probe_chans = [1,4,5,6], # 7 is good, but takes up space.
average = False,  #  in future, make this an option
#srange = range(60, 92),   # 196ms range over which to create frames (or to average)
#srange = range(200, 222),   # towards end before rise (2000,2)
srange = range(230, 248),   # during fall (2000,2) 508MS
minpts=16,
)))
for ss in [51]: db.update({'LP20160309_{ss}'.format(ss=ss): deepcopy(db['LP20160309_52'])})
#db['LP20160309_51'].update(dict(Te_range=[10,50], p_range=[0,1.5], srange=range(160,180)))

db.update(dict(LP20160224_31=dict(
Te_range = [8, 40],  # [8, 60] for 51
ne_range = [0, 10],
#probe_chans = [1,4,5,6,7],
probe_chans = [1,4,5,6],
average = False,  #  in future, make this an option
srange = range(60,80), # 92),   # range over which to create frames (or to average)
#srange = range(200, 222),   # towards end before rise (2000,2)
#srange = range(230, 262),   # towards end during rise (2000,2)
minpts=16,
)))
for ss in [23, 26, 30, 35]: db.update({'LP20160224_{ss}'.format(ss=ss): deepcopy(db['LP20160224_31'])})
db['LP20160224_26'].update(dict(Te_range=[10,50], p_range=[0,1.5], srange=range(160,180)))
db['LP20160224_30'].update(dict(Te_range=[10,60], p_range=[0,1.5], srange=range(160,180)))

db.update(dict(LP20160308_23=dict(
Te_range = [8, 80],  # [8, 60] for 51
ne_range = [0, 5],
#probe_chans = [1,4,5,6,7],
probe_chans = [1,4,5,6],
average = False,  #  in future, make this an option
srange = range(60,80),
minpts=16,
)))
for ss in [22,24]: db.update({'LP20160308_{ss}'.format(ss=ss): deepcopy(db['LP20160308_23'])})

"""
# dafile = 'LP20160224_25_L53'
dafile = 'LP20160224_25_L53_2k2.npz'
dafile = 'LP20160224_25_L57_2k2.npz'
Te_range = [10, 51]  # both  # 53 [10,70]    57 [10,100]
ne_range = [0, 10]   # both  # 53 [0,12]      57 [0,10]
srange = range(85, 95)   # 20160224_25_L53  0.22 (0.4)
srange = range(145, 175)   # 20160224_25_L53  0.33
srange = range(155, 205)   # 20160224_25_L53  0.4
srange = range(145,160) #  0224_25
#srange = range(175, 225)   # 20160224_25_L53  0.4
minpts=18


dafile = 'LP20160309_41_L57_2k2.npz'
Te_range = [10, 80]  # 
ne_range = [0, 6]   # 
srange = range(145, 175)   # 20160308_38_L53  0.33
srange = range(105, 135)   # 20160224_25_L53  0.33

minpts=18
"""

#########################
_var_defaults = """
dafile = 'foo'
dummy = 0  # True will generate dummy data
NGY = 600j  # number of points in image grid
NGX = NGY # theoretically need more res in X, but
          # near horizontal lines at top and bottom need more Y
#rc('font', **{'size':18})
minpts = 17
probe_chans = [1]
loc = 'best'
marg = 0.2  # room to leave at either end of the time axis
srange = range(110, 130)
p_range = [0, 1]  # .4 for 224:35 - default
start_time = None # 0.9 # None  # default
#srange = range(len(da['t_mid']))
step = 20  # 3  # use every (step) time slice but skip unsuitable ones
ne_scl_basic = 500  # 500 good for normal sized plots - will be normalised later

# Soren says seg 7,  9 and 19 are damaged, and seg 3 10 and 12
suppress_ne = '3_LP10::3_LP11::7_LP09::7_LP11::7_LP12::7_LP19'.split('::')
show_power = 0
ne_kwargs = dict(cmap='jet')   # 'gray_r'
nrm = [None, None]

"""

exec(_var_defaults)

# read single arg if there are no equal signs - otherwise interpret options
if len(sys.argv)>1 and np.all(['=' not in sa for sa in sys.argv]):
    if len(sys.argv)>1:
        dafile=sys.argv[1]

    if len(sys.argv)>2:
        start_time=float(sys.argv[2])

    if len(sys.argv)>3:
        minpts=int(sys.argv[3])

else:
    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())


#if not(os.path.exists(dafile)):
#    print('try LP/ folder')
#    dafile = 'LP/' + dafile

try:
    if da.name == dafile:
        print('NOT reloading file - run without -i to force reload')
    else:
        raise NameError
except NameError:
    print('temp fix')
    # da = DA(dafile)

# if the filename matches anythin in the dbm read those values in
dbkey = os.path.splitext(os.path.split(dafile)[1])[0].split('_L5')[0]
locals().update(db[dbkey])

areas = 'uncalibrated'

if show_power:
    ne_range = p_range if p_range is not None else [0, 1]
    ne_lab = 'Pdens\n$MW/m^{2}$'
else:
    ne_lab = '$n_e/10^{18}$\n$m^{-3}$'

sc_kwargs = dict(vmin=Te_range[0], vmax=Te_range[1]) if Te_range is not None else {}
ne_kwargs.update(dict(vmin=ne_range[0], vmax=ne_range[1]) if ne_range is not None else {})


st = 0
skipped = []
figs= []
num = None  # None auto numbers figures


figsize=(10, 7) if len(figs)>0 else plt.rcParams['figure.figsize']  # first figure is rc default size
fig, (axu, axl) = plt.subplots(2, 1, sharex=True, num=num, figsize=np.array(figsize)[::-1])

# see N2_puff_correlation for the vertical offsets of text point labels
for ax, seg in zip([axu, axl],['3','7']):

    da = DA(dafile.replace('SEG',seg))
    if (len(da['Te'][0])<minpts):
        raise LookupError('fewer channels ({nc}) than minpts'.
                          format(nc = len(da['Te'][0])))
    try:
        if 'params' in da['info']:
            if  da['info']['params']['pyfusion_version'] > '0.6.7b':
                areas = 'approximate'
            else:
                print('*** Warning - incorrect limiter numbers? ***')
    except:
        print('******* Really Old data file??***********')

    # take care of obsolete names
    ne = 'ne18' if 'ne18' in da.da else 'ne'
    Vf = 'Vf' if 'Vf' in da.da else 'Vp'
    # ne_max is used to offset the labels according to the size of the dots - (don't want them shifting from frame to frame)
    ne_max = np.nanmax(da.masked[ne],0)
    wnan = np.where(np.isnan(ne_max))[0]
    ne_max[wnan] = 0
    lowerz = 0.165
    upperz = 0.25
    st = 0

    if start_time is not None:
        srange = np.where((da['t']>start_time) & (da['t']<start_time+.03))[0]
        print('srange = {sr}'.format(sr=srange))

    for s in srange:
        ne_raw = da.masked[ne][s]
        print(len(ne_raw)), 
        Te_raw = da.masked['Te'][s]
        eTe_raw = da['eTe'][s] if 'eTe' in da else 10 + 0*Te_raw
        Vf_raw  = da.masked[Vf][s]
        I0_raw  = da.masked['I0'][s]
        if nrm[seg == '7'] is not None:
            ne_raw_orig = ne_raw * 1.0
            ne_raw = ne_raw/nrm[seg == '7']
            I0_raw = I0_raw*ne_raw/ne_raw_orig/nrm[seg == '7']

        #  ne_cleaned has the suspect channels removed (see suppress_ne)

        coords = np.array(da.infodict['coords'])
        X, Y, Z = np.array(coords).T
        # print(Z)
        th = np.deg2rad(-18)
        x = X * np.cos(th) - Y*np.sin(th)
        y = X * np.sin(th) + Y*np.cos(th)
        z = Z

        if show_power:
            import json
            thispath = os.path.dirname(__file__)
            W7X_path = os.path.realpath(thispath+'/../acquisition/W7X/')
            fname = os.path.join(W7X_path,'limiter_geometry.json')
            limdata = json.load(open(fname))

            slp = np.polyval(np.polyder(limdata['shape_poly']),np.abs(x))
            sintheta = -np.sin(np.arctan(slp))
            ne_cleaned = I0_raw * (Vf_raw + 4 * Te_raw) * sintheta
            ne_raw = ne_cleaned.copy()
        else:
            ne_cleaned = ne_raw.copy()

        for (c, ch) in enumerate(da.infodict['channels']):
            if np.any([suppr in ch for suppr in suppress_ne]):
                ne_cleaned[c] = np.nan
        wg = np.where(~np.isnan(ne_raw))[0]
        if len(wg) < minpts: #  Too few - get next
            print('{n} is too few (see minpts)'.format(n=len(wg)))
            continue
        if (step > 0):
            if st == 0:
                print('s={s}'.format(s=s)),
                good_list = wg
                ne_list = []
                Te_list = []
                eTe_list = []
                Vf_list = []
                I0_list = []
                t_mid_list = []
                tslice = None
            st += 1
            if st <= step and not np.any(np.isnan(ne_raw[good_list])):  # ne is the most likely to get wiped
                #    skipped.append(s)
                print('grabbed {s}'.format(s=s)),

                t_mid_list.append(da['t_mid'][s])
                # print(t_mid_list)
                ne_list.append(ne_raw)
                Te_list.append(Te_raw)
                eTe_list.append(eTe_raw)
                Vf_list.append(Vf_raw)
                I0_list.append(I0_raw)
            if st < step: 
                print('cont',s,st),
                continue
            else:
                div = np.sum([1/err for err in eTe_list],0)
                ne_raw = np.sum([n/err for n,err in zip(ne_list, eTe_list)],0)/div  # wghtd avg ne, Te, Vf
                Te_raw = np.sum([Te/err for Te,err in zip(Te_list, eTe_list)],0)/div
                Vf_raw = np.sum([VF/err for VF,err in zip(Vf_list, eTe_list)],0)/div
                I0_raw = np.sum([Is/err for Is,err in zip(I0_list, eTe_list)],0)/div
                # can't use this technique for a scalar
                tslice = np.average(t_mid_list)
                st=0  # reset
                ck_ch=6
                print('\n\n*** Averaged over {n} frames, at {t}. Te for channel {c} = {Te:.1f} \n{fn}'
                      .format(n=len(eTe_list), c=da['info']['channels'][wg[ck_ch]], 
                              t=tslice, Te=Te_raw[ck_ch], fn=da.name))

        # fig = plt.figure(100, figsize=(12, 8))
        # if len(figs) > 5:  # if more than 5, reuse figure 100
        #    num = 100
        # figsize=(8, 6) if len(figs)>0 else plt.rcParams['figure.figsize']  # first figure is default size
        # fig, (axu, axl) = plt.subplots(2, 1, sharex=True, num=num, figsize=np.array(figsize)[::-1])
        # fig.set_figheight(1.5*fig.get_figheight())  # no effect!
        # fig.set_size_inches(fig.get_size_inches()[::-1], forward=True) # this works I think
        figs.append(fig)
        time_strip_h = 0.16  # 0.2  0.12
        split = 0.03
        broken_axes = 0
        fig.subplots_adjust(bottom=.11 + time_strip_h, left=0.13, hspace=split, right=0.75, top=0.93)
        if time_strip_h>0:
            axtime = fig.add_axes([0.08, 0.06, 0.9-0.08, time_strip_h])

        if broken_axes:
            axu.spines['bottom'].set_visible(False)
            axl.spines['top'].set_visible(False)
            axu.xaxis.tick_top()
            axu.tick_params(labeltop='off')  # don't put tick labels at the top
            axl.xaxis.tick_bottom()

            dia = .015  # how big to make the diagonal lines in axes coordinates
            dy = .02 if split==0 else 0

            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=axu.transAxes, color='k', clip_on=False)
            axu.plot((-dia, dia), (dy-dia, dy+dia), **kwargs)        # top-left diagonal
            axu.plot((1 - dia, 1 + dia), (dy-dia, dy+dia), **kwargs)  # top-right diagonal

            kwargs.update(transform=axl.transAxes)  # switch to the bottom axes
            axl.plot((-dia, dia), (1 - dy - dia, 1 -dy + dia), **kwargs)  # bottom-left diagonal
            axl.plot((1 - dia, 1 + dia), (1 - dy - dia, 1 -dy + dia), **kwargs)  # bottom-right diagonal
        # end if broken_axes
        ne_scl = fig.get_size_inches()[0]/8.0 * ne_scl_basic
        if ne_range is not None:
            ne_scl = ne_scl/ne_range[1]  # normalize ne_scl

        print('{nf} figs, last index is {l}'
              .format(nf=len(figs), l=len(da['t_mid'])))
        wg_suppr = np.where(~np.isnan(ne_cleaned))[0]
        wg_dodgy = np.lib.arraysetops.setdiff1d(wg,wg_suppr)  # dodgy refers to suppressed
        ne_cleaned = ne_cleaned[wg_suppr]
        Te_raw = da.masked['Te'][s]
        if dummy:
            x, z = np.mgrid[-.07:.07:21j, .15:.25:23j]
            x, z = x.flatten(), z.flatten()
            ne_raw = np.cos(10*x)*np.exp(3*z)

        coords2D = np.array(zip(x, z))
        sgn = int(np.sign(np.nanmean(z)))
        grid_x, grid_z = np.mgrid[-.06:.06:NGX, lowerz*sgn:upperz*sgn:NGY]
        # 'nearest', 'linear', 'cubic'
        negr = griddata(coords2D[wg_suppr], ne_cleaned, (grid_x, grid_z), method='cubic')
        org = 'lower' if (sgn > 0) else 'upper'
        ext = (np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z))
        axim = ax.imshow(negr.T, origin=org, aspect='equal', extent=ext, **ne_kwargs)
        shr = 0.1  #  the colorbars are a little shorter
        cbarneax = plt.axes([0.88, 0.1 + shr + time_strip_h, 0.04, 0.8 - time_strip_h - 2*shr])
        cbarne = plt.colorbar(axim,cax=cbarneax)
        # cbarne.set_label(r'$n_e/10^{18}$', rotation=270, labelpad=15, fontsize='large')
        cbarne.ax.title.set_text(ne_lab)
        cbarne.ax.title.set_fontsize('large')
        #cbarne.ax.set_xlabel(r'$n_e/10^{18}$', fontsize='large') # ignored ,verticalalignment='top')
        w = wg_dodgy  # dodgy points first
        dodgy_kwargs = sc_kwargs.copy()
        dodgy_kwargs.update(dict(linestyle=':',lw=10,edgecolor='gray'))
        sp = ax.scatter(x[w], z[w], ne_scl*ne_raw[w], Te_raw[w], **dodgy_kwargs)
        # now better ones
        w = wg_suppr
        Temappable = ax.scatter(x[w], z[w], ne_scl*ne_cleaned, Te_raw[w], **sc_kwargs)
        cbarTeax = plt.axes([0.79, 0.1 + shr + time_strip_h, 0.05, 0.8 - time_strip_h - 2*shr])
        cbarTe = plt.colorbar(Temappable, cax=cbarTeax)
        #  cbarTe.set_label(r'$T_e (eV)$', rotation=270, fontsize='large')
        cbarTe.ax.title.set_text('$T_e$\n$(eV)$')
        cbarTe.ax.title.set_fontsize('large')
        #cbarTe.ax.set_xlabel(r'$T_e (eV)$', fontsize='large')
        # from here on wg is selected
        ne_OK = ne_raw[wg]
        Te_neOK = Te_raw[wg]
        x = x[wg]
        z = z[wg]
        ax.plot([0, 0], ax.get_ylim(), linewidth=.3)  # do first so it is a little shorter
        ax.set_ylim(np.sort([sgn*(lowerz-.005), sgn*(upperz+.005)]))

        locator = MaxNLocator(prune='upper', nbins=6) # trim nbins to keep numbers on x axis round
        ax.xaxis.set_major_locator(locator)
        plt.show()
        for (c, ch) in enumerate(np.array(da.infodict['channels'])[wg]):
            lab = ch[2:]
            lab = ch[4:]+'$^'+ch[2]+'$'
            ax.text(x[c] + (2e-4*np.sqrt(ne_scl*ne_max[c]) + .001)*np.array([1,-1])[x[c]<0], z[c],
                      lab, fontsize='x-small', horizontalalignment=['left','right'][x[c]<0])
                      #  ch[2:], fontsize='x-small', horizontalalignment=['left','right'][x[c]<0])

    #figs[-1].suptitle('W7-X limiter 5 seg {seg} amu {amu} (point color is Te, size is ne: {areas} probe areas)'
    #                  .format(areas=areas, seg=[0,3,7][sgn],amu=da.infodict['params'].get('amu','?')))
    axu.set_title('W7-X limiter 5 Langmuir Array, segments 3 and 7 '#amu={amu}'
                  .format(areas=areas, seg=[0,3,7][sgn],amu=da.infodict['params'].get('amu','?')))

    ax.set_ylabel('Z (m)')
    axl.set_xlabel('X (m)')
    titl = str('shaded area - {lab}\nsize of dots - {lab}\n'\
               'color of dots - $T_e$'
               .format(lab=ne_lab.split('\n')[0]))
    leg=axl.legend((),title=titl,
                   loc = 'center', bbox_to_anchor = (0.05, 1.02))
    leg.draggable()

dev_name = 'W7X'
dev = pyfusion.getDevice(dev_name)
shot = da['info']['shotdata']['shot'][0]
chan = da['info']['channels'][0]
echdata = dev.acq.getdata(shot,'W7X_TotECH')
wech = np.where(echdata.signal > 100)[0]
tech = echdata.timebase[wech[0]]
t0_utc = int(tech * 1e9) + echdata.utc[0]
dtprobe = (da['info']['params']['i_diag_utc'][0] - t0_utc)/1e9
tslice = da['t_mid'][s-step//2] + dtprobe
dtaverage = da['t_mid'][s] - da['t_mid'][s-step]
dtstr = '{dt:.2f}s avg'.format(dt=dtaverage)
# axu.set_title('{fn}, time={t:.4f} {dt}'.format(fn=da.name, t=tslice, dt=dtstr))
tit = axu.get_title()
cfg, cfg_dict = get_mag_config(shot)
if cfg is None:
    ind = ''
else:
    ind = 13 - 12*cfg[2]/.3904
    if abs(ind - round(ind)) < .03:
        ind = int(round((ind)))
    ind = 'Ind {ind}'.format(ind=ind)

axu.set_title(tit+'\nProgram Id {s}, time={t:.4f} {ind}'# {dt}'
              .format(s=shot, fn=da.name, t=tslice, dt=dtstr, ind=ind))

if cfg_dict is not None:
    plot_trim(axu, mag_dict=cfg_dict, color='g', aspect=1.2)

if time_strip_h > 0:
    locator = MaxNLocator(nbins=3)  # , prune='upper') # This is for the time

    #probedata = dev.acq.getdata(shot,'W7X'+chan + '_I')
    from pyfusion.data.filters import dummysig
    # plot ECH til after probes so colours match up
    diags = ['ne18','Te']
    axtwin = axtime.twinx()
    deflw = plt.rcParams['lines.linewidth']
    for (i, diag) in enumerate(diags):
        for (pp, prch) in enumerate(probe_chans):
            probedata = dummysig(da['t_mid'],da[diag][prch])
            probedata.signal = da.masked[diag][:,prch]  # kludge 
            probedata.utc = da['info']['params']['i_diag_utc']
            dtprobe = (probedata.utc[0] - t0_utc)/1e9
            axx = axtime if i == 0 else axtwin
            axx.plot(probedata.timebase + dtprobe, probedata.signal,
                     ls=['-', ':'][i], lw=[1*deflw, 1.5*deflw][i], # dashes heavier than line
                     label='{diag} s{ch}'  # _ needs to have sepcial treatment if full latex
                     .format(diag=diag, ch=da['info']['channels'][prch][2:]))

    axtime.plot(echdata.timebase - tech, echdata.signal/1000, label='ECH',lw=1.5*plt.rcParams['lines.linewidth'])
    gasdata = dev.acq.getdata(shot, 'W7X_GasCtlV_23')
    dtgas = (gasdata.utc[0] - t0_utc)/1e9
    wplasmagas = np.where((gasdata.timebase+dtgas > np.min(probedata.timebase+dtprobe)) 
                          & (gasdata.timebase+dtgas < np.max(probedata.timebase+dtprobe)))[0]
    if np.max(gasdata.signal[wplasmagas]) > 0.1:
        axtime.plot(gasdata.timebase + dtgas, gasdata.signal,label=gasdata.config_name[4:])
    axtime.set_xlim(-0.01,max(probedata.timebase + dtprobe))
    ## cleanp = probedata.signal[np.where(~np.isnan(probedata.signal))[0]]
    ## probemax = (np.sort(cleanp))[int(0.98*len(cleanp))] if len(cleanp)>10 else ne_range[1]
    #axtime.set_ylim(0,2*np.nanmean(probedata.signal))
    #axtime.set_ylim(0, 1.05*max(probemax, np.max(echdata.signal)/1000))
    # Can't get probemax density as Te was the last plotted
    probemax = axtime.get_ylim()[1] if show_power else ne_range[1]
    axtime.set_ylim(0, 1.05*max(probemax, np.max(echdata.signal)/1000))
    gaspuff = get_puff(shot, t_range = axtime.get_xlim())
    if gaspuff is not None and np.max(gaspuff[1])>0.05:
        useax = axtwin if np.max(gaspuff[1]) > axtime.get_ylim()[1] else axtime
        useax.plot(gaspuff[0], gaspuff[1], label=gaspuff[2])
    axtime.yaxis.set_major_locator(locator)


    axtime.plot([tslice,tslice],axtime.get_ylim(),'k',lw=2)
    twlocator = MaxNLocator(nbins=3, prune='upper') # reduce clutter on twin
    axtwin.set_ylim(0,50)
    axtwin.yaxis.set_major_locator(twlocator)

    act_xlim = axtime.get_xlim()
    dif = act_xlim[1] - act_xlim[0]
    axtime.set_xlim(act_xlim[0] - marg*dif, act_xlim[1] + marg*dif)


    legt = axtime.legend(prop={'size':'x-small'},loc=loc)
    if legt: 
        legt.draggable()

    if len(diags) > 1:
        leg = axtwin.legend(prop={'size':'x-small'},loc=2)
        if leg:  # this is not draggable (at least when they are on top of each other)
            leg.draggable()

    axtime.set_ylabel('$n_e$')
    axtwin.set_ylabel('$T_e$')
    axtime.set_xlabel('time (s)')

if len(srange)/float(step) > 4:
    root, ext = os.path.splitext(da.name)
    path, name = os.path.split(root)
    folder = os.path.join(path,'movie')
    figs[-1].savefig(os.path.join(folder,'{fn}_{fr:03d}'
                                  .format(folder=folder, fn=name, fr=s)))
    plt.close(figs[-1])
else:
    plt.show()

numOK = [len(np.where(~np.isnan(slic))[0]) for slic in da.masked['Te']]
firstOK = np.where(np.array(numOK)>8)[0][0]
print('first reasonably complete time is t_mid = {t:.4f}, index {f}'.format(t=da['t_mid'][firstOK], f=firstOK))

