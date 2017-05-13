#!/usr/bin/env python
""" 

Short form:
sql_plot kappa_h,ne18_bar mrk=o  # ne vs k_h for the last day or 100 shots
# the above uses defaults for _where, and _order
# you can then select suitable ones 
   _where='shot between 1000 and 2000'  # need quotes for spaces
   _order='kapp_s'   # the order_by is added automatically if omitted

Note: need to quote spaces carefully - also plkw if using the dict() form
run ~/python/sql_plot.py _select='select kappa_ABB,ne18_bmax' _where='where shot between 101000 and 200000' label='shot' "plkw=dict(color='g')"
_PYFUSION_TEST_@@block=0
"""
from sqlsoup import SQLSoup
from sqlite3 import OperationalError
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np
import sys
sys.path.append('/home/bdb112/python')
from bdb_utils.process_cmd_line_args import process_cmd_line_args

_var_defaults="""
_select = 'select shot, kappa_V, mirnov_RMS'
_from = 'from summary_h1'
# this tricky _where gets a nice default set - more complicated than normal
_where = 'where date(recorded) = date() or shot > (select max(shot) from summary_h1) -100'
_order = ''
_group = ''
_limit = ''
mrk = '-'
idx = None  # defaults to [0,1]
label=''  # label each point (of the first series) with this value
maxlabels=100
db_url='sqlite:////data/summary.db' # or 
#db_url='sqlite:////rmt/h1svr/home/datasys/virtualenvs/h1ds_production/h1ds/h1ds/db/summary.db' 
split=5 # if >0, split lines according to the column which has the fewest unique values - if split=5, only do it is there are 5 unique values or less.
def __help__():
    print('local help for sqlplot')
fig=None
plkw={} #  "plkw=dict(color='g')"
hold=0
block=1
"""
exec(_var_defaults)
if len(sys.argv) >= 2 and '=' not in sys.argv[1]:
    _select = sys.argv.pop(1)

exec(process_cmd_line_args())

def __help__(): # can't make this work.
    print('local help for sqlplot')

### all this is tricky stuff to allow for brief queries
# fix up common mistakes - leaving out select or where
_select = 'select ' + _select if not 'select' in _select else _select
_where = 'where ' + _where if not 'where' in _where else _where
_order = 'order by ' + _order if len(_order)>1 and not 'order' in _order else _order

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

if label is not '':
    _select += ', ' + label
qry = ' '.join([_select, _from, _where, _order, _group, _limit])
try:
    res = h1db.execute(qry)
except (OperationalError, Exception) as reason:
    print(reason)
    if 'no such column' in reason.message:
        var = reason.message.split(': ')[1].split()[0]
        from difflib import SequenceMatcher
        matches = np.argsort([SequenceMatcher(None, col, var).ratio() for col in cols])
        print('\nInstead of {v} try {c}'
              .format(v=var, c = [cols[m] for m in matches[::-1][:10]]))
    import sys
    sys.exit()

xx = res.fetchall()
xxt = np.array(xx).T
# put them in a dictionary
# note that the presence of Nones requires fiddling to avoid 'objects'
dat = OrderedDict()

for (k, key) in enumerate(xx[0].keys()):
    vals =  xxt[k]
    bads = [v is None for v in vals]
    #vals[[np.where(bads)[0]]] = np.nan
    dat.update({key: np.array(vals.tolist(),dtype=[None, int][key=='shot'])})

labels = label.split(',')
keys = list(dat)
for lab in labels[::-1]:  # must remove in reverse order of adding to find them at the end
    if keys[-1] == lab: # remove the key of the label 
        keys.pop(-1)

idx = range(len(keys)) if idx is None else idx
ykeys = [keys[i] for i in idx[1:]] if len(idx)>1 else idx

# plot the data in separate lines if one of the vars is in groups
# look for the key with the fewest unique values - (and less then <split> - say 5)
num_uniqs = [len(np.unique(dat[key])) for key in ykeys]
if min(num_uniqs) < split and len(ykeys)>1:
    gkey = ykeys[np.argmin(num_uniqs)]
    inds_list = [np.where(dat[gkey] == uval)[0] for uval in np.unique(dat[gkey])]
    label_extras = [': {gkey} = {v:.3g}'.format(gkey=gkey, v=uval) for uval in np.unique(dat[gkey])]
    idx = [i for i in idx if keys[i] != gkey]
else:   # just one range
    inds_list = [range(len(dat[keys[idx[0]]]))]
    label_extras = ['']

if fig is not None:
    plt.figure(fig)

# label_extras refers to the legend, label, labels refer to the point labels.
for p,(inds,labex) in enumerate(zip(inds_list, label_extras)):
    for pp,iidx in enumerate(idx[1:]):
        plt.plot(dat[keys[idx[0]]][inds], dat[keys[iidx]][inds], mrk, 
                 label=keys[iidx]+labex, hold=(hold or p>0 or pp>0), **plkw)
        print(p),

        if label is not '':
            for ind in inds[:maxlabels]:    # can't print too many
                labtext = '\n'.join([' {}'.format(dat[lab][ind]) for lab in labels 
                                        if  not np.isnan(dat[keys[iidx]][ind])])
                plt.text(dat[keys[idx[0]]][ind], dat[keys[iidx]][ind], labtext, fontsize='x-small', verticalalignment='center')

plt.xlabel(keys[idx[0]])
# split qry into nice lnegth lines
titl = ''
for wd in qry.split(' '):
    if len(titl.split('\n')[-1] + wd)>80:
        titl += '\n' + wd
    else:
        titl += ' ' + wd
plt.title(titl, fontsize='medium')
plt.legend(prop=dict(size='small'))
plt.show(block)
