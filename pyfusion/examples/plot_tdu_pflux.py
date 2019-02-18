"""
Extract the prelim_anal probe data from tar files and plot a number of day's worth
\\sv-e4-fs-1\E4-Mitarbeiter\E4 Diagnostics\QRP_Langmuir_Sonden\OP1.2_TDU-Sonden\Preliminary_Analysis

maxopen=0 is supposed to prevent any screen graphics, but it makes punny saved plots and makes screen graphics anyway
"""

from __future__ import print_function           
import os
import tarfile
import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.cm as cm
from pyfusion.utils.process_cmd_line_args import process_cmd_line_args
from datetime import date
from dateutil.rrule import rrule, DAILY

all_dates = [dt.strftime("%Y%m%d") for dt in rrule(DAILY, dtstart=date(2018, 7, 1), until=date(2018, 10, 19))]


def showfig(fig, savefile='', suptitle=None, maxopen=None):
    if maxopen is None:
        maxopen = 20 if savefile is '' else 2

    if maxopen > 0:  # don't waste time showing if no
        print('showing', maxopen)
        fig.canvas.manager.full_screen_toggle()
        fig.show()
    fig.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, y=.997)
        topmarg = 1 -fig.subplotpars.top
        fig.subplots_adjust(top=1 -1.2*topmarg, wspace=0.02)
    if savefile is not '':
        fig.savefig(savefile)

match_text = 'LD'
diag = 'pf'
dec = 10 
dates = ['20180724']
path = '/data/datamining/local_data/W7X/prelim_anal'
savefmt = '{date}_{match_text}_{diag}.png'
maxopen = 3  # will close fig if there are more than this
lw = plt.rcParams['lines.linewidth']
suptitle = 'Parallel power flux (MW/m2)'
maxperpage = 60
image = True
inset_pos = 1 # 2: 'upper left' 1:Up R,  3:LL 4: LowR  # loc='best' hangs forever
exec(process_cmd_line_args())

all_figs = []
for date in dates:
    timefile = '{p}/times/{d}_times.tgz'.format(d=date, p=path)
    datfile = '{p}/{diag}/{dt}_{diag}.tgz'.format(diag=diag, dt=date, p=path)
    if not os.path.exists(datfile):
        print('skipping ' + date, end=' ') 
        continue
    tart = tarfile.open(timefile.format(d=date, p=path))
    tarp = tarfile.open(datfile)
    numgood = len([f for f in tarp if match_text in f.name and not np.isnan(np.nanmax(np.loadtxt(tarp.extractfile(f))))])
    numperpage = min(numgood + 1, maxperpage)
    nrows = int(np.sqrt(numperpage))
    ncols = min([cs for cs in range(3 * nrows) if nrows * cs >= (numperpage + 1)])
    savefile = savefmt.format(date=date, match_text=match_text, diag=diag)
    axiter = iter([])
    if maxopen == 0:
        plt.interactive(False)
    figs = []
    for tf in tart:
        shot_text = tf.name.split('times')[1]
        pfnames = [fn for fn in tarp.getnames() if shot_text in fn and match_text in fn]
        if len(pfnames) == 0:
            continue
        if len(pfnames) > 1:
            print('more than one match to {st}, {mt} found'.format(st=shot_text, mt=match_text))
        pfname = pfnames[0]
        t = np.loadtxt(tart.extractfile(tf))
        pf = np.loadtxt(tarp.extractfile(pfname))
        if np.isnan(np.nanmax(pf)):
            print('all nan', pfname)
            continue
        if len(t) != len(pf[0]):
            print('time mismatch', pfname)
            continue
        
        y201, rest = pfname.split('201')
        shot = int(rest.split(diag)[1].split('.')[0])
        prog = '201' + rest[0:5] + '.' + str(shot)
        try:
            ax = next(axiter, None)
            if ax is None:
                if len(figs) > 0:
                    showfig(figs[-1], savefile=savefile, suptitle=suptitle, maxopen=maxopen)
                fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharey='all', sharex='all', squeeze=False, figsize=[24,16])
                figs.append(fig)
                if len(all_figs) > maxopen + 1:  # don't pop the last one, otherwise the loop will restart
                    print('all_figs', [f.number for f in all_figs])
                    if 1 or hasattr(all_figs[0],'close'):
                        cf = all_figs.pop(0)
                        print('closing ' + str(cf.number))
                        plt.close(cf)
                    #all_figs.pop(0).close()
                axiter = iter([axx for axrow in axs for axx in axrow])  # flatten
                ax = next(axiter)

            scl = 1e6 if diag in ['pf','pPerp'] else -1e-3 if 'Sat' in diag else 1.
            if image:
                ax.imshow(pf, aspect='auto', cmap=cm.jet)
            else:
                ax.step(1 + np.arange(len(pf)), np.nanmean(pf.T, axis=0) / scl, where='mid')
            
            # 2: 'upper left' 1(Up R) 3:LL 4: LowR  # loc='best' hangs forever
            axins = inset_axes(ax, width="40%", height="40%", loc=inset_pos,
                               borderpad=0.3, axes_kwargs=dict(alpha=0.5))
            my_cycler = (cycler('color', 'b g r c m y k b g r'.split()) +
                         cycler('lw', 7*[lw] + 3*[1.5*lw]))  # the last 3 repeat but thicker
            axins.set_prop_cycle(my_cycler)
            axins.locator_params(nbins=5)
            axins.plot(t[::dec], pf.T[::dec]/scl)
            # axins.yaxis.set_ticklabels([])
            #  https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.tick_params.html
            axins.text(1, 0.99, prog, verticalalignment='top',
                       horizontalalignment='right', transform=axins.transAxes, fontsize='xx-small')
            axins.xaxis.set_tick_params(labelsize='x-small')
            lleft = image==True;
            lright = not image
            axins.yaxis.set_tick_params(labelsize='x-small', labelleft=lleft, labelright=lright, right=lright, left=lleft)
            # axins.set_ylim(ax.get_ylim())
            ax.set_title(pfname.split('.')[0])
        except Exception as reason:
            print('Exception {r} in {fn}'.format(r=reason, fn=pfname))
    ax = next(axiter, ax)  # if there is no room left, plop the legend on the last graph!
    ax.set_prop_cycle(my_cycler)
    # show the colours and lines in the last window
    for p in range(len(pf)):
        ax.plot([p, p], label='LP{n:02}'.format(n=p + 1))
        ax.legend(prop={'size': 'x-small'}, ncol=int(np.sqrt(len(pf) / 2.)))
        
    showfig(figs[-1], savefile=savefile, suptitle=suptitle, maxopen=maxopen)
    all_figs.extend(figs)

"""

dates = ['20180724']
path = '/data/datamining/local_data/W7X/prelim_anal'

timefile = '{p}/times/{d}_times.tgz'.format(d=date, p=path)
pPerpfile = '{p}/{diag}/{dt}_{diag}.tgz'.format(diag='pPerp', dt=date, p=path)
Tefile = '{p}/{diag}/{dt}_{diag}.tgz'.format(diag='Te', dt=date, p=path)
nefile = '{p}/{diag}/{dt}_{diag}.tgz'.format(diag='ne', dt=date, p=path)
tart = tarfile.open(timefile.format(d=date, p=path))
tarp = tarfile.open(datfile)

t = np.loadtxt(tart.extractfile(tf))
pf = np.loadtxt(tarp.extractfile(pfname))

"""
