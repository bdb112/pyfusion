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

dafile = '20160302_12_L57'
Te_range = None
ne_range = None

dafile = '20160302_12_L57'
Te_range = [10, 100]  # 20160302_12
ne_range = [0, 2]

dafile = '20160302_12_L57'
Te_range = [10, 100]  # 20160302_12
ne_range = [0, 2]

dafile = '20160310_9_L57'
Te_range = [20, 100]  # 
ne_range = [0, 10]

dafile = '20160310_9_L53'
Te_range = [20, 100]  # 
ne_range = [0, 20]
minpts=18

"""
dafile = '20160310_39_L57'
Te_range = [10, 100]  # 
ne_range = [0, 4]


dafile = '20160308_41_L57' # also 44
Te_range = [10, 100]  # 
ne_range = [0, 6]

"""
dafile = '20160309_42_L57'
Te_range = [10, 100]  # 53 [10,70]
ne_range = [0, 10]   # 53 [0,10]
minpts=18

da = DA('LP/' + dafile)
areas = 'uncalibrated'
try:
    if 'params' in da['info']:
        if  da['info']['params']['pyfusion_version'] > '0.6.7b':
            areas = 'approximate'
except:
    print('Old data file??')

sc_kwargs = dict(vmin=Te_range[0], vmax=Te_range[1]) if Te_range is not None else {}
ne_kwargs = dict(vmin=ne_range[0], vmax=ne_range[1]) if ne_range is not None else {}

if ne_range is not None:
    ne_scl = 500./ne_range[1]

srange = range(100, 200)
srange = range(180, 185)
srange = range(67, 75)   # 20160309_42_L53
srange = range(110, 111)
#srange = range(len(da['t_mid']))
step = 3  # 3  # use every (step) time slice but skip unsuitable ones

st = 0
skipped = []
figs= []
num = None  # None auto numbers figures
for s in srange:
    ne = 'ne18' if 'ne18' in da.da else 'ne'
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
    grid_x, grid_z = np.mgrid[-.06:.06:NGX, .16*sgn:.25*sgn:NGY]
    # 'nearest', 'linear', 'cubic'
    ne = griddata(coords2D, ne_raw, (grid_x, grid_z), method='cubic')
    org = 'lower' if (sgn > 0) else 'upper'
    axim = plt.imshow(ne.T, origin=org, extent=(np.min(grid_x), np.max(grid_x), np.min(grid_z), np.max(grid_z)),
                      **ne_kwargs)
    ax = axim.get_axes()
    cbarne = plt.colorbar(fraction=0.08, pad=0.01)
    # cbarne.set_label(r'$n_e/10^{18}$', rotation=270, labelpad=15, fontsize='large')
    cbarne.ax.set_xlabel(r'$n_e/10^{18}$', fontsize='large')
    plt.scatter(x, z, ne_scl*ne_raw, Te_raw, **sc_kwargs)
    plt.plot([0, 0], ax.get_ylim(), linewidth=.3)
    locator = MaxNLocator(prune='upper')
    ax.xaxis.set_major_locator(locator)
    for (c, ch) in enumerate(np.array(da.infodict['channels'])[wg]):
        plt.text(x[c]+[.004,-0.018][x[c]<0], z[c], ch[2:], fontsize='x-small')
    
    cbarTe = plt. colorbar(fraction=0.08, pad=0.02)
    #  cbarTe.set_label(r'$T_e (eV)$', rotation=270, fontsize='large')
    cbarTe.ax.set_xlabel(r'$T_e (eV)$', fontsize='large')
    figs[-1].suptitle('W7-X limiter sect 5 seg {seg} amu {amu} (point size is ne, shade is Te: {areas} probe areas)'
                      .format(areas=areas, seg=[0,7,3][sgn],amu=da.infodict['params'].get('amu','?')))
    strip_h = 0.12
    plt.subplots_adjust(bottom=.1 + strip_h, left=0.05, right=1)
    axstr = figs[-1].add_axes([0.145,0.05,0.68,strip_h])
    locator = MaxNLocator(nbins=3)  # , prune='upper')

    dev_name = 'W7X'
    dev = pyfusion.getDevice(dev_name)
    shot = da['info']['shotdata']['shot'][0]
    chan = da['info']['channels'][0]
    #probedata = dev.acq.getdata(shot,'W7X'+chan + '_I')
    from pyfusion.data.filters import dummysig
    probedata = dummysig(da['t_mid'],da['ne18'][0])
    probedata.signal = da.masked['ne18'][:,0]  # kludge 
    probedata.utc = da['info']['params']['i_diag_utc']
    
    echdata = dev.acq.getdata(shot,'W7X_TotECH')
    wech = np.where(echdata.signal > 100)[0]
    tech = echdata.timebase[wech[0]]
    t0_utc = int(tech * 1e9) + echdata.utc[0]
    dtprobe = (probedata.utc[0] - t0_utc)/1e9
    axstr.plot(probedata.timebase + dtprobe, probedata.signal)
    axstr.plot(echdata.timebase - tech, echdata.signal/1000)

    gasdata = dev.acq.getdata(shot,'W7X_GasCtlV_23')
    dtgas = (gasdata.utc[0] - t0_utc)/1e9
    axstr.plot(gasdata.timebase + dtgas, gasdata.signal)
    axstr.yaxis.set_major_locator(locator)
    axstr.set_xlim(-0.01,max(probedata.timebase + dtprobe))
    axstr.set_ylim(0,2*np.nanmean(probedata.signal))

    tslice = da['t_mid'][s] + dtprobe
    axstr.plot([tslice,tslice],axstr.get_ylim(),'k',lw=2)
    ax.set_title('{fn}, time={t:.4f}'.format(fn=da.name, t=tslice))

    if len(srange) > 4:
        root, ext = os.path.splitext(da.name)
        path, name = os.path.split(root)
        folder = os.path.join(path,'movie')
        figs[-1].savefig(os.path.join(folder,'{fn}_{fr:03d}'
                                      .format(folder=folder, fn=name, fr=s)))
        plt.close(figs[-1])
    else:
        plt.show()

