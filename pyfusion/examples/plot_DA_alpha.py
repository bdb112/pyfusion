"""  overlay different LP analysis windows with various alpha based on 
rough estimate of uncertainty"""
import numpy as np
from matplotlib import pyplot as plt
from pyfusion.utils.process_cmd_line_args import process_cmd_line_args
from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup
from pyfusion.data.DA_datamining import Masked_DA, DA

dafile = 'pmulti_20181010_31_52_1_1.05'
verbose=0
area_scale = None
nerr_bin_edges = [0, .2, .4,  .7, 1, 2, 5]
nerr_scale = 0.5
areas =     [4,  2, .5, .2, .1, .05]
areas =     [3,  2, 1, .5, .2, .1]
cols = ['r','g','b']
lw_scale = 1
marker='|'
intrp = 1
tstart = 1.0
tend = 1.05
lw = 0.5
period = 20e-6

exec(process_cmd_line_args())

da = DA(dafile, load=1, verbose=verbose)
tm, = da.extract(varnames='t_mid')
idt = np.argsort(da['t_mid'])
tm, = da.extract(varnames='t_mid', inds=idt)

area_scale = np.sqrt(len(tm))/10 if area_scale is None else area_scale
nerr_bin_edges = nerr_scale * np.array(nerr_bin_edges) 

fig, axs = plt.subplots(3, 1, sharex='all', squeeze=True)
for k, key in enumerate('Te:I0:Vf'.split(':')):
    val, err = da.extract(varnames='{k},e{k}'.format(k=key), inds=idt)
    wgood = np.where(~np.isnan(val) & ~np.isnan(err))[0]
    val = val[wgood]
    err = err[wgood]
    tmg = tm[wgood]
    # needs to be unsigned, also Vf is a problem, how to know average
    absavg = np.average(np.abs(val), weights=1 / (0.1 + err))
    if key == 'Vf':
        absavg *= 3
    nerr = np.clip(err/absavg, 0.05, 100)
    print('{k}: weighted ave = {wa:.2g}, avg = {av:.2g}'
          .format(k=key, wa=absavg, av=np.average(val)))
    ax = axs[k]
    if intrp:
        nbins = (tend - tstart) / period
        tbins = np.linspace(tstart, tend, nbins+1, endpoint=True)
        bedges = tmg.searchsorted(tbins)
        # xrange may be faster, should be smaller
        groups = [range(bedges[bb],bedges[bb+1]) for bb in range(len(bedges)-1)]
        valintavg = [np.average(val[gp], weights=nerr[gp])
                     for g, gp in enumerate(groups) if len(gp)>0]
        nerrintavg = [np.average(nerr[gp], weights=nerr[gp])
                     for g, gp in enumerate(groups) if len(gp)>0]
        timintavg = [np.average([tbins[g],tbins[g+1]])
                     for g, gp in enumerate(groups) if len(gp)>0]
        # ax.plot((tint[0:-1] + tint[1:])/2, valint, color=cols[k], lw=lw)
        # ax.plot(timintavg, valintavg, color=cols[k], lw=lw)
        # now linearly interp over gaps - valreg
        valreg = np.interp(tbins, timintavg, valintavg)
        timreg = np.interp(tbins, timintavg, timintavg)
        nerrreg = np.interp(tbins, timintavg, nerrintavg)
        for nerrmax, thk in zip(nerr_bin_edges[1:][::-1], areas[::-1]):
            wbad = np.where(nerrreg > nerrmax)[0]
            print('{n} below {l}'.format(n=len(nerrreg) - len(wbad), l=nerrmax))
            valtmp = np.array(valreg)
            valtmp[wbad] = np.nan
            ax.plot(timreg, valtmp, color=cols[k], lw=lw * thk)

    else:
        for e, upper in enumerate(nerr_bin_edges[1:]):
            lower = nerr_bin_edges[e - 1]
            wbin = np.where((nerr > lower) & (nerr < upper))[0]
            print('{n} below {l}'.format(n=len(wbin), l=lower))
            if len(wbin) > 0:
                if lw_scale is not None:
                    ax.plot(tmg[wbin], val[wbin], lw=areas[e]/lw_scale, color=cols[k])
                else:
                    ax.plot(tmg[wbin], val[wbin], markersize=areas[e]/area_scale, color=cols[k], marker=marker, ls='')
    
    axs[k].set_ylabel(key)
fig.show()
