""" Script to animate Te and ne - first attempt - needs generalising
"""
#_PYFUSION_TEST_@@Skip
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
from pyfusion.data.DA_datamining import DA
import pyfusion

from matplotlib.ticker import MaxNLocator

dummy = 0  # True will generate dummy data
NGX = 300j  # number of points in image grid
NGY = 2*NGX
#rc('font', **{'size':18})
minpts = 17
probe_chans = [1]
loc = 'best'
srange = range(110, 120)
#srange = range(len(da['t_mid']))
step = 3  # 3  # use every (step) time slice but skip unsuitable ones

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


dafile = 'LP20160309_10_L57_amoeba21_1.2_2k.npz'
Te_range = [10, 70]  
ne_range = [0, 3]   
srange = range(60, 72)   # early High power region
srange = range(200, 212)   # mid lower power
minpts=18
probe_chans = [1,6]

dafile = 'LP20160309_52_L57_amoeba21_1.2_2k.npz'
Te_range = [0, 50]  
ne_range = [0, 15]
probe_chans = [1,6]
srange = range(60, 72)   # 
#srange = range(200, 212)   # towards end before rise (2000,2)
#srange = range(240, 252)   # towards end during rise (2000,2)
minpts=18

"""
dafile = 'LP20160224_25_L53'
Te_range = [10, 50] #both  # 53 [10,70]    57 [10,100]
ne_range = [0, 10]  #both  # 53 [0,12]      57 [0,10] 
srange = range(85, 95)   # 20160224_25_L53
minpts=18
"""

da = DA('LP/' + dafile)
areas = 'uncalibrated'
try:
    if 'params' in da['info']:
        if  da['info']['params']['pyfusion_version'] > '0.6.7b':
            areas = 'approximate'
        else:
            print('*** Warning - incorrect limiter numbers? ***')
except:
    print('******* Really Old data file??***********')

sc_kwargs = dict(vmin=Te_range[0], vmax=Te_range[1]) if Te_range is not None else {}
ne_kwargs = dict(vmin=ne_range[0], vmax=ne_range[1]) if ne_range is not None else {}

if ne_range is not None:
    ne_scl = 500./ne_range[1]



st = 0
skipped = []
figs= []
num = None  # None auto numbers figures

ne = 'ne18' if 'ne18' in da.da else 'ne'
# ne_max is used to offset the labels according to the size of the dots - (don't want them shifting in time)
ne_max = np.nanmax(da.masked[ne],0)
wnan = np.where(np.isnan(ne_max))[0]
ne_max[wnan] = 0
lowerz = 0.16
upperz = 0.25

for s in srange:
    ne_raw = da.masked[ne][s]
    wg = np.where(~np.isnan(ne_raw))[0]
    st += 1
    if (step>0 and st > 1) or len(wg) < minpts:
        if st <= 1:
            skipped.append(s)
            st -= 1
        if st >= step: st=0  # reset
        continue
    #fig = plt.figure(100, figsize=(12, 8))
    if len(figs)>5: # if more than 5, reuse figure 100 
        num = 100
    figs.append (plt.figure(num, figsize=(8, 6)))
    ne_raw = ne_raw[wg]
    Te_raw = da.masked['Te'][s][wg]
    coords = np.array(da.infodict['coords'])[wg]
    X, Y, Z = np.array(coords).T
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
    negr = griddata(coords2D, ne_raw, (grid_x, grid_z), method='cubic')
    org = 'lower' if (sgn > 0) else 'upper'
    axim = plt.imshow(negr.T, origin=org, aspect='equal',
                      extent=(np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z)), **ne_kwargs)
    ax = axim.get_axes()
    cbarne = plt.colorbar(fraction=0.08, pad=0.01)
    # cbarne.set_label(r'$n_e/10^{18}$', rotation=270, labelpad=15, fontsize='large')
    cbarne.ax.set_xlabel(r'$n_e/10^{18}$', fontsize='large')
    sp = ax.scatter(x, z, ne_scl*ne_raw, Te_raw, **sc_kwargs)
    ax.plot([0, 0], ax.get_ylim(), linewidth=.3)
    ax.set_ylim(sgn*(lowerz-.005), sgn*(upperz+.005))

    locator = MaxNLocator(prune='upper')
    ax.xaxis.set_major_locator(locator)
    for (c, ch) in enumerate(np.array(da.infodict['channels'])[wg]):
        plt.text(x[c] + (2e-4*np.sqrt(ne_scl*ne_max[c]) + .001)*np.array([1,-1])[x[c]<0], z[c], ch[2:],
                 fontsize='x-small', horizontalalignment=['left','right'][x[c]<0])
    
    cbarTe = plt.colorbar(sp, fraction=0.08, pad=0.02)
    #  cbarTe.set_label(r'$T_e (eV)$', rotation=270, fontsize='large')
    cbarTe.ax.set_xlabel(r'$T_e (eV)$', fontsize='large')
    figs[-1].suptitle('W7-X limiter sect 5 seg {seg} amu {amu} (point size is ne, shade is Te: {areas} probe areas)'
                      .format(areas=areas, seg=[0,3,7][sgn],amu=da.infodict['params'].get('amu','?')))
    strip_h = 0.2
    plt.subplots_adjust(bottom=.1 + strip_h, left=0.05, right=1)
    axtime = figs[-1].add_axes([0.105,0.05,0.72,strip_h])
    locator = MaxNLocator(nbins=3)  # , prune='upper')

    dev_name = 'W7X'
    dev = pyfusion.getDevice(dev_name)
    shot = da['info']['shotdata']['shot'][0]
    chan = da['info']['channels'][0]
    #probedata = dev.acq.getdata(shot,'W7X'+chan + '_I')
    from pyfusion.data.filters import dummysig
    echdata = dev.acq.getdata(shot,'W7X_TotECH')
    wech = np.where(echdata.signal > 100)[0]
    tech = echdata.timebase[wech[0]]
    t0_utc = int(tech * 1e9) + echdata.utc[0]
    axtime.plot(echdata.timebase - tech, echdata.signal/1000, label='ECH')
    for prch in probe_chans:
        probedata = dummysig(da['t_mid'],da['ne18'][prch])
        probedata.signal = da.masked['ne18'][:,prch]  # kludge 
        probedata.utc = da['info']['params']['i_diag_utc']
        dtprobe = (probedata.utc[0] - t0_utc)/1e9
        axtime.plot(probedata.timebase + dtprobe, probedata.signal,
               label='ne18 '+ da['info']['channels'][prch][4:])

    gasdata = dev.acq.getdata(shot,'W7X_GasCtlV_23')
    dtgas = (gasdata.utc[0] - t0_utc)/1e9
    wplasmagas = np.where((gasdata.timebase+dtgas > np.min(probedata.timebase+dtprobe)) 
                          & (gasdata.timebase+dtgas < np.max(probedata.timebase+dtprobe)))[0]
    if np.max(gasdata.signal[wplasmagas]) > 0.1:
        axtime.plot(gasdata.timebase + dtgas, gasdata.signal,label=gasdata.config_name[4:])
    axtime.yaxis.set_major_locator(locator)
    axtime.set_xlim(-0.01,max(probedata.timebase + dtprobe))
    #axtime.set_ylim(0,2*np.nanmean(probedata.signal))
    axtime.set_ylim(ne_range)

    tslice = da['t_mid'][s] + dtprobe
    axtime.plot([tslice,tslice],axtime.get_ylim(),'k',lw=2)
    ax.set_title('{fn}, time={t:.4f}'.format(fn=da.name, t=tslice))
    plt.legend(prop={'size':'x-small'},loc=loc)

    if len(srange)/float(step) > 4:
        root, ext = os.path.splitext(da.name)
        path, name = os.path.split(root)
        folder = os.path.join(path,'movie')
        figs[-1].savefig(os.path.join(folder,'{fn}_{fr:03d}'
                                      .format(folder=folder, fn=name, fr=s)))
        plt.close(figs[-1])
    else:
        plt.show()

