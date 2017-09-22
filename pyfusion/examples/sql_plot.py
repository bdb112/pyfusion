#!/usr/bin/env python
""" 

Short form:
sql_plot kappa_h,ne18_bar mrk=o  # ne vs k_h for the last day or 100 shots
# the above uses defaults for _where, and _order
# you can then select suitable ones
   _where='shot between 1000 and 2000'  # need quotes for spaces
   _order='kapp_s'   # the order_by is added automatically if omitted

Note: need to quote spaces carefully - also plkw if using the dict() form
run ~/python/sql_plot.py _select='select kappa_ABB,ne18_bmax' _where='where shot between 101000 and 200000' plabel='shot' "plkw=dict(color='g')"
_PYFUSION_TEST_@@block=0

Example showing the swap feature - plot selected columns in kind of alphabetical order labelled with the first column (kappa_V)
>> run pyfusion/examples/sql_plot.py 'kappa_V, \*' "_from=combsum" '_where="shot between 92647 and 92658"' swap=1

plotting with channels on X axis (swapwild)

run pyfusion/examples/sql_plot.py 'kappa_V,\*' "_from=combsum" '_where="shot between 97490 and 97580 and round(kh_req,2) is .3 and rfp_req is 40"' swapwild='ne[0-9]*$' sub=[2,4]

# a join on the fly
run pyfusion/examples/sql_plot.py kh_req,kappa_V,mirnov_RMS,m02y,m04y,m06y,m08y,m10y,m12y,m13y,m14y,m16y,m05x,m06x,m02x " _where=  (summary_h1.shot between 93405 and 93437)" _from="from summary_h1 join mirnov_RMS using (shot)" mrk='o-/s-/^-'

"""
from sqlsoup import SQLSoup
from sqlite3 import OperationalError
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np
import sys
import re

sys.path.append('/home/bdb112/python')
from bdb_utils.process_cmd_line_args import process_cmd_line_args

_var_defaults = """
_select = 'select shot, kappa_V, mirnov_RMS'
_from = 'from summary_h1'
# this tricky _where gets a nice default set - more complicated than normal
_where = 'where date(recorded) = date() or shot > (select max(shot) from summary_h1) -100'
_order = ''
_group = ''
_limit = ''
mrk = '-'  # can be a string using / as a separator (, is a valid mark)
idx = None  # defaults to [0,1]
plabel=''  # label each point (of the first series) with this value
maxplabels=300
db_url='sqlite:////data/summary.db' # or 
#db_url='sqlite:////rmt/h1svr/home/datasys/virtualenvs/h1ds_production/h1ds/h1ds/db/summary.db' 
split=5 # if >0, split lines according to the column which has the fewest unique values - if split=5, only do it is there are 5 unique values or less.
def __hxelp__():
    print('local help routine for sqlplot, defined in _var_defaults')
fig=None
plkw={} #  "plkw=dict(color='g')"
colors='' # '' means normal python sequence
colors='b,g,r,c,m,y,k,gray,orange,purple,pink,brown,lightgreen'
labels=''  # '' means get from sql, labels=_,_ will suppress the first two
           # or in matplotlib 2, _nolegend_,_nolegend_ will suppress them
sub=None   # the position of a substring in the short name to use in sorting for plotting [3,4] or 3
hold=0
block=0
swapwild=''    # '^m.*[x,y,z]'  # if not empty, plot matching columns on the x axis (sorted)
clip = None # if not None clip all values to this range
"""
# grab the first arg if it does not contain an "="
exec(_var_defaults)
if len(sys.argv) >= 2 and '=' not in sys.argv[1]:
    _select = sys.argv.pop(1)  # and pop it out


def __help__():  # must be before exec() line
    print('local help routine for sqlplot!')


exec(process_cmd_line_args())
colors = colors.split(',') if ',' in colors else colors
labels = labels.split(',') if ',' in labels else labels

### all this is tricky stuff to allow for brief queries
# fix up common mistakes - leaving out select or where
_select = 'select ' + _select if 'select' not in _select else _select
_where = 'where ' + _where if 'where' not in _where else _where
_order = 'order by ' + _order if len(_order)>1 and 'order' not in _order else _order
_from = 'from ' + _from if len(_from)>1 and 'from' not in _from else _from

# if we asked for one variable, assume it is plotted against shot
vars = _select.lower().split('select ')[1].split(',')
if len(vars) == 1:
    _select = _select.replace('select', 'select shot,')

# default order is by shot
if _order is '':
    if len(vars) is 1:
        _order = 'order by shot'
    else:
        _order = 'order by ' + vars[0]

h1db = SQLSoup(db_url)
cols = h1db.summary_h1.c.keys()
h1db.SUMM = h1db.summary_h1

if plabel is not '':
    _select += ', ' + plabel
qry = ' '.join([_select, _from, _where, _order, _group, _limit])
try:
    res = h1db.execute(qry)
except (OperationalError, Exception) as reason:
    print(reason)
    if 'no such column' in reason.message:
        # won't work for combsum - SQLSoupError: table 'combsum' does not have a primary key defined
        var = reason.message.split(': ')[1].split()[0]
        from difflib import SequenceMatcher
        matches = np.argsort([SequenceMatcher(None, col, var).ratio() for col in cols])
        print('\nInstead of {v} try {c}'
              .format(v=var, c=[cols[m] for m in matches[::-1][:10]]))
    import sys
    sys.exit()

xx = res.fetchall()
xxt = np.array(xx).T
# put them in a dictionary
# note that the presence of Nones requires fiddling to avoid 'objects'
dat = OrderedDict()

if len(xx) == 0:
    raise LookupError('no data matching\n' + qry)

for (k, key) in enumerate(xx[0].keys()):
    vals = xxt[k]
    bads = [v is None for v in vals]
    # vals[[np.where(bads)[0]]] = np.nan
    dat.update({key: np.array(vals.tolist(), dtype=[None, int]['shot' in key])})

if swapwild != '':  # plot columns across x axis
    matcher = re.compile(swapwild)
    sub = [sub, sub+1] if (sub is not None) and (not isinstance(sub, (list, tuple, np.ndarray))) else sub 
    names = np.sort([n for n in list(dat) if matcher.match(n)])
    sorted_names = np.array(names)[np.argsort([s[sub[0]:sub[1]] for s in names], kind='mergesort')] if sub is not None else names
    xy = [dat[x] for x in sorted_names]
    if len(sorted_names) < 2:
        raise ValueError("Can't graph with just 1 point\n" + qry)

    xyT = np.array(xy, dtype=float).T  # for contours etc
    xyT = xyT.clip(*clip) if clip is not None else xyT
    fig, (ax_xy, ax_im) = plt.subplots(2, 1, sharex='none',
                                       gridspec_kw=dict(hspace=0.25))
    sw_plt = ax_xy.plot(xy)
    # xticks(range(1, 1+len(sorted_names)), sorted_names)
    allnames = list(dat)
    othernames = [n for n in allnames if n not in sorted_names]

    for y, val in zip(dat[sorted_names[-1]], dat[othernames[0]]):
        if y is not None and not np.isnan(y):
            ax_xy.text(ax_xy.get_xlim()[1], y, str(val), horizontalalignment='right')

    # can make extent [0.5, len() + 0.5... to centre ticks on boxes, but that confuses xticks
    extent = [0, len(names), dat[othernames[0]][0], dat[othernames[0]][-1]]
    matim_axim = ax_im.imshow(xyT, aspect='auto', interpolation='none',
                              origin='lower', extent=extent)
    ax_xy.set_xticks(np.array(range(len(names))))
    ax_im.set_xticks(0.5 + np.array(range(len(names))))
    ax_im.set_ylabel('y axis not linear')
    for ax in [ax_xy, ax_im]:
        ax.set_xticklabels(sorted_names, rotation=90, fontsize='small')
    plt.colorbar(matim_axim, ax=(ax_im, ax_xy))

    plt.show()
    sys.exit()  # plt.figure()

plabels = plabel.split(',')
keys = list(dat)
for lab in plabels[::-1]:  # must remove in reverse order of adding to find them at the end
    if keys[-1] == lab:    # remove the key of the plabel
        keys.pop(-1)

idx = range(len(keys)) if idx is None else idx
ykeys = [keys[i] for i in idx[1:]] if len(idx)>1 else idx

# plot the data in separate lines if one of the vars is in groups
# look for the key with the fewest unique values - (and less then <split> - say 5)
# ignore Nulls and nans when looking for uniques
num_uniqs = [len(np.unique([dd for dd in dat[key] if not dd is None and not np.isnan(dd)])) for key in ykeys]
num_uniqs = [nu if nu > 0 else np.nan for nu in num_uniqs]  # set then number to nan if it is 0 (useless key)
if np.nanmin(num_uniqs) < split and len(ykeys)>1:  # and so we have to use nanmin etc
    gkey = ykeys[np.nanargmin(num_uniqs)]
    inds_list = [np.where(dat[gkey] == uval)[0] for uval in np.unique(dat[gkey])]
    label_extras = [': {gkey} = {v:.3g}'.format(gkey=gkey, v=uval) if uval is not None else ': '+gkey+' = None' for uval in np.unique(dat[gkey])]
    idx = [i for i in idx if keys[i] != gkey]
else:   # just one range
    inds_list = [range(len(dat[keys[idx[0]]]))]
    label_extras = ['']

if keys[idx[0]] == 'rowid':
    dat[keys[idx[0]]] = np.arange(len(dat[keys[idx[0]]]))
    
if fig is not None:
    plt.figure(fig)
else:
    fig = plt.gcf()

lines = []  # prepare line list for legend picker
    
# label, label_extras refers to the legend, plabel, plabels refer to the point labels.
grouped_shots = []
for p,(inds,labex) in enumerate(zip(inds_list, label_extras)):
    grouped_shots.append([dat['shot'][ii] for ii in inds])
    # mrk can be a comma, so split using / instead
    mk = mrk.split('/')[p] if '/' in mrk else mrk
    for pp,iidx in enumerate(idx[1:]):
        col = colors[pp] if pp < len(colors) else None
        label = labels[p] if p < len(labels) else keys[iidx]+labex
        lin, = plt.plot(dat[keys[idx[0]]][inds], dat[keys[iidx]][inds], mk, color=col,
                        label=label,  hold=(hold or p>0 or pp>0), **plkw)
        lines.append(lin)
        print(p),

        if plabel is not '':
            # for ind in inds[:maxplabels]:    # can't print too many
            for ind in inds[::max(1,len(inds)//maxplabels)]:    # can't print too many
                yval = dat[keys[iidx]][ind]
                labtext = ''
                for lab in plabels:
                    if yval is not None and not np.isnan(yval):
                        if len(labtext) > 0:
                            labtext += '\n'
                        labtext += '  ' + str(dat[lab][ind])  # space on left

                plt.text(dat[keys[idx[0]]][ind], dat[keys[iidx]][ind], labtext,
                         fontsize='x-small', verticalalignment='center')

plt.xlabel(keys[idx[0]])
# split qry into nice lnegth lines
titl = ''
for wd in qry.split(' '):
    if len(titl.split('\n')[-1] + wd)>80:
        titl += '\n' + wd
    else:
        titl += ' ' + wd
plt.title(titl, fontsize='medium')

# implement the legend picker from https://matplotlib.org/examples/event_handling/legend_picking.html
leg = plt.legend(prop=dict(size='small'),loc='best')
# leg.draggable()  # conflicts with on/off picker

lined = dict()
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = origline


def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

# if there aleady is a callback, don't confuse by adding another
# This is relevant if the script is run again
if len(fig.canvas.callbacks.callbacks.get('pick_event',[])) < 1:
    fig.canvas.mpl_connect('pick_event', onpick)

plt.show(block)
