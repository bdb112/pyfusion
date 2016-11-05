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

dummy = 0  # True will generate dummy data
NGY = 600j  # number of points in image grid
NGX = NGY # theoretically need more res in X, but
          # near horizontal lines at top and bottom need more Y
#rc('font', **{'size':18})
minpts = 17
probe_chans = [1]
loc = 'best'
srange = range(110, 120)
#srange = range(len(da['t_mid']))
step = 14  # 3  # use every (step) time slice but skip unsuitable ones
ne_scl_basic = 500  # 500 good for normal sized plots - will be normalised later

# Soren says seg 7,  9 and 19 are damaged, and seg 3 10 and 12
suppress_ne = '3_LP10::3_LP11::7_LP09::7_LP11::7_LP12::7_LP19'.split('::')

ne_kwargs = dict(cmap='jet')   # 'gray_r'

dafile = '20160302_12_L57'
Te_range = None
ne_range = None

dafile = '20160302_12_L57'
Te_range = [10, 100]  # 20160302_12
ne_range = [0, 2]

dafile = 'LP20160302_12_L57'
Te_range = [10, 100]  # 20160302_12
ne_range = [0, 2]

dafile = 'LP20160310_9_L57'
Te_range = [20, 100]  # 
ne_range = [0, 10]


dafile = 'LP20160310_9_L53'
Te_range = [20, 100]  # 
ne_range = [0, 20]
minpts=18


dafile = 'LP20160310_39_L57'
Te_range = [10, 100]  # 
ne_range = [0, 4]


dafile = 'LP20160308_41_L57' # also 44
Te_range = [10, 100]  # 
ne_range = [0, 6]

dafile = 'LP20160309_42_L53'
Te_range = [10, 50]  # 53 [10,70]
ne_range = [0, 10]   # 53 [0,10]
srange = range(60, 72)   # 20160309_42_L53
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

Te_range = [8, 60]  # [8, 60] for 51
ne_range = [0, 15]
probe_chans = [1,4,5,6,7]
probe_chans = [1,4,5,6]
average = False  #  in future, make this an option
srange = range(60, 92)   # range over which to create frames (or to average)
#srange = range(200, 222)   # towards end before rise (2000,2)
#srange = range(230, 262)   # towards end during rise (2000,2)
minpts=18


# dafile = 'LP20160224_25_L53'
dafile = 'LP20160224_25_L53_2k2.npz'
dafile = 'LP20160224_25_L57_2k2.npz'
Te_range = [10, 51]  # both  # 53 [10,70]    57 [10,100]
ne_range = [0, 10]   # both  # 53 [0,12]      57 [0,10]
srange = range(85, 95)   # 20160224_25_L53  0.22 (0.4)
srange = range(145, 175)   # 20160224_25_L53  0.33
srange = range(155, 205)   # 20160224_25_L53  0.4
#srange = range(175, 225)   # 20160224_25_L53  0.4
minpts=18
"""

dafile = 'LP20160309_41_L57_2k2.npz'
Te_range = [10, 80]  # 
ne_range = [0, 6]   # 
srange = range(145, 175)   # 20160308_38_L53  0.33
srange = range(105, 135)   # 20160224_25_L53  0.33

minpts=18
"""
if len(sys.argv)>1:
    dafile=sys.argv[1]

if len(sys.argv)>2:
    minpts=int(sys.argv[2])

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

areas = 'uncalibrated'

sc_kwargs = dict(vmin=Te_range[0], vmax=Te_range[1]) if Te_range is not None else {}
ne_kwargs.update(dict(vmin=ne_range[0], vmax=ne_range[1]) if ne_range is not None else {})


st = 0
skipped = []
figs= []
num = None  # None auto numbers figures


srange=[145,160]
figsize=(8, 6) if len(figs)>0 else plt.rcParams['figure.figsize']  # first figure is rc default size
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


    ne = 'ne18' if 'ne18' in da.da else 'ne'
    # ne_max is used to offset the labels according to the size of the dots - (don't want them shifting from frame to frame)
    ne_max = np.nanmax(da.masked[ne],0)
    wnan = np.where(np.isnan(ne_max))[0]
    ne_max[wnan] = 0
    lowerz = 0.165
    upperz = 0.25

    for s in srange:
        ne_raw = da.masked[ne][s]
        Te_raw = da.masked['Te'][s]
        eTe_raw = da['eTe'][s]
        #  ne_cleaned has the suspect channels removed (see suppress_ne)
        ne_cleaned = ne_raw.copy()
        for (c, ch) in enumerate(da.infodict['channels']):
            if np.any([sup in ch for sup in suppress_ne]):
                ne_cleaned[c] = np.nan
        wg = np.where(~np.isnan(ne_raw))[0]
        if len(wg) < minpts: #  Too few - get next
            print('{n} is too few (see minpts)'.format(n=len(wg)))
            continue
        step=2
        if (step > 0):
            if st == 0:
                print(s),
                ne_list = []
                Te_list = []
                eTe_list = []
            st += 1
            if st <= step:
                #    skipped.append(s)
                ne_list.append(ne_raw)
                Te_list.append(Te_raw)
                eTe_list.append(eTe_raw)
            if st < step: 
                print('cont')
                continue
            else:
                div = np.sum([1/err for err in eTe_list],0)
                ne_raw = np.sum([n/err for n,err in zip(ne_list, eTe_list)],0)/div
                Te_raw = np.sum([Te/err for Te,err in zip(Te_list, eTe_list)],0)/div
                st=0  # reset
                ck_ch=6
                print('Averaged over {n} frames, Te for channel {c} = {Te:.1f}'
                      .format(n=len(eTe_list), c=da['info']['channels'][wg[ck_ch]], Te=Te_raw[ck_ch]))
        # fig = plt.figure(100, figsize=(12, 8))
        # if len(figs) > 5:  # if more than 5, reuse figure 100
        #    num = 100
        # figsize=(8, 6) if len(figs)>0 else plt.rcParams['figure.figsize']  # first figure is default size
        # fig, (axu, axl) = plt.subplots(2, 1, sharex=True, num=num, figsize=np.array(figsize)[::-1])
        # fig.set_figheight(1.5*fig.get_figheight())  # no effect!
        # fig.set_size_inches(fig.get_size_inches()[::-1], forward=True) # this works I think
        figs.append(fig)
        time_strip_h = 0.12  # 0.2
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
        wg_sup = np.where(~np.isnan(ne_cleaned))[0]
        wg_dodgy = np.lib.arraysetops.setdiff1d(wg,wg_sup)  # dodgy refers to suppressed
        ne_cleaned = ne_cleaned[wg_sup]
        Te_raw = da.masked['Te'][s]
        coords = np.array(da.infodict['coords'])
        X, Y, Z = np.array(coords).T
        print(Z)
        th = np.deg2rad(-18)
        x = X * np.cos(th) - Y*np.sin(th)
        y = X * np.sin(th) + Y*np.cos(th)
        z = Z
        if dummy:
            x, z = np.mgrid[-.07:.07:21j, .15:.25:23j]
            x, z = x.flatten(), z.flatten()
            ne_raw = np.cos(10*x)*np.exp(3*z)

        coords2D = np.array(zip(x, z))
        sgn = int(np.sign(np.nanmean(z)))
        grid_x, grid_z = np.mgrid[-.06:.06:NGX, lowerz*sgn:upperz*sgn:NGY]
        # 'nearest', 'linear', 'cubic'
        negr = griddata(coords2D[wg_sup], ne_cleaned, (grid_x, grid_z), method='cubic')
        org = 'lower' if (sgn > 0) else 'upper'
        ext = (np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z))
        axim = ax.imshow(negr.T, origin=org, aspect='equal', extent=ext, **ne_kwargs)
        shr = 0.1  #  the colorbars are a little shorter
        cbarneax = plt.axes([0.9, 0.1 + shr + time_strip_h, 0.04, 0.8 - time_strip_h - 2*shr])
        cbarne = plt.colorbar(axim,cax=cbarneax)
        # cbarne.set_label(r'$n_e/10^{18}$', rotation=270, labelpad=15, fontsize='large')
        cbarne.ax.title.set_text('$n_e/10^{18}$\n$m^{-3}$')
        cbarne.ax.title.set_fontsize('large')
        #cbarne.ax.set_xlabel(r'$n_e/10^{18}$', fontsize='large') # ignored ,verticalalignment='top')
        w = wg_dodgy  # dodgy points first
        dodgy_kwargs = sc_kwargs.copy()
        dodgy_kwargs.update(dict(linestyle=':',lw=10,edgecolor='gray'))
        sp = ax.scatter(x[w], z[w], ne_scl*ne_raw[w], Te_raw[w], **dodgy_kwargs)
        # now better ones
        w = wg_sup
        Temappable = ax.scatter(x[w], z[w], ne_scl*ne_raw[w], Te_raw[w], **sc_kwargs)
        cbarTeax = plt.axes([0.8, 0.1 + shr + time_strip_h, 0.05, 0.8 - time_strip_h - 2*shr])
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

    leg=axl.legend((),title='shaded area - $n_e$\nsize of dots - $n_e$\ncolor of dots - $T_e$')
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
axu.set_title(tit+'\nProgram Id {s}, time={t:.4f}'# {dt}'
              .format(s=shot, fn=da.name, t=tslice, dt=dtstr))

if time_strip_h > 0:
    locator = MaxNLocator(nbins=3)  # , prune='upper') # This is for the time

    #probedata = dev.acq.getdata(shot,'W7X'+chan + '_I')
    from pyfusion.data.filters import dummysig
    # plot ECH til after probes so colours match up
    diags = ['ne18','Te']
    axtwin = axtime.twinx()
    for (i, diag) in enumerate(diags):
        for (pp, prch) in enumerate(probe_chans):
            probedata = dummysig(da['t_mid'],da[diag][prch])
            probedata.signal = da.masked[diag][:,prch]  # kludge 
            probedata.utc = da['info']['params']['i_diag_utc']
            dtprobe = (probedata.utc[0] - t0_utc)/1e9
            axx = axtime if i == 0 else axtwin
            axx.plot(probedata.timebase + dtprobe, probedata.signal,
                     ls=['-', ':'][i], lw=[1, 2][i],
                     label='{diag} s{ch}'  # _ needs to have sepcial treatment if full latex
                     .format(diag=diag, ch=da['info']['channels'][prch][2:]))

    axtime.plot(echdata.timebase - tech, echdata.signal/1000, label='ECH')
    gasdata = dev.acq.getdata(shot, 'W7X_GasCtlV_23')
    dtgas = (gasdata.utc[0] - t0_utc)/1e9
    wplasmagas = np.where((gasdata.timebase+dtgas > np.min(probedata.timebase+dtprobe)) 
                          & (gasdata.timebase+dtgas < np.max(probedata.timebase+dtprobe)))[0]
    if np.max(gasdata.signal[wplasmagas]) > 0.1:
        axtime.plot(gasdata.timebase + dtgas, gasdata.signal,label=gasdata.config_name[4:])
    axtime.set_xlim(-0.01,max(probedata.timebase + dtprobe))
    #axtime.set_ylim(0,2*np.nanmean(probedata.signal))
    axtime.set_ylim(ne_range)

    gaspuff = get_puff(shot, t_range = axtime.get_xlim())
    if gaspuff is not None:
        useax = axtwin if np.max(gaspuff[1]) > axtime.get_ylim()[1] else axtime
        useax.plot(gaspuff[0], gaspuff[1], label=gaspuff[2])
    axtime.yaxis.set_major_locator(locator)


    axtime.plot([tslice,tslice],axtime.get_ylim(),'k',lw=2)
    twlocator = MaxNLocator(nbins=3, prune='upper') # reduce clutter on twin
    axtwin.set_ylim(0,50)
    axtwin.yaxis.set_major_locator(twlocator)
    legt = axtime.legend(prop={'size':'x-small'},loc=loc)
    if legt: 
        legt.draggable()

    if len(diags) > 1:
        leg = axtwin.legend(prop={'size':'x-small'},loc=loc)
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

