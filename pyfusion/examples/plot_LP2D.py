""" Script to animate Te and ne - first attempt - needs generalising
"""
#_PYFUSION_TEST_@@Skip
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyfusion.data.DA_datamining import DA

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

dafile = '20160310_9_L53'
Te_range = [10, 100]  # 
ne_range = [0, 2]

dafile = '20160310_39_L57'
Te_range = [10, 100]  # 
ne_range = [0, 4]

dafile = '20160308_41_L57' # also 44
Te_range = [10, 100]  # 
ne_range = [0, 6]


dafile = '20160309_42_L53'
Te_range = [10, 70]  # 
ne_range = [0, 8]
minpts=18

da = DA(dafile)

sc_kwargs = dict(vmin=Te_range[0], vmax=Te_range[1]) if Te_range is not None else {}
ne_kwargs = dict(vmin=ne_range[0], vmax=ne_range[1]) if ne_range is not None else {}

if ne_range is not None:
    ne_scl = 500./ne_range[1]

srange = range(100, 200)
srange = range(180, 185)
srange = range(67, 75)   # 20160309_42_L53
#srange = range(110, 111)
srange = range(len(da['t_mid']))
step = 0  # 3  # use every (step) time slice but skip unsuitable ones

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
        plt.text(x[c], z[c], '    '+ch[2:], fontsize='xx-small')
    cbarTe = plt. colorbar(fraction=0.08, pad=0.02)
    #  cbarTe.set_label(r'$T_e (eV)$', rotation=270, fontsize='large')
    cbarTe.ax.set_xlabel(r'$T_e (eV)$', fontsize='large')
    plt.title('{fn}, time={t:.4f}'.format(fn=da.name, t=da['t_mid'][s]))
    figs[-1].suptitle('W7-X limiter sect 5 seg {seg} amu {amu} (point size is ne, shade is Te: uncalibrated probe areas)'
                      .format(seg=[0,7,3][sgn],amu=da.infodict['params'].get('amu','?')))
    plt.subplots_adjust(left=0.05, right=1)
    if len(srange) > 4:
        figs[-1].savefig('{folder}{fn}_{fr:03d}'.format(folder='movie/',fn=da.name.split('.')[0], fr=s))
        plt.close(figs[-1])
    else:
        plt.show()

