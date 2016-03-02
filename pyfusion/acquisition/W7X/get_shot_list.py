#!/usr/bin/env python
""" By extracting all progId, comments and dates, in ProjectDesc,
create a table to connect shots, dates and times.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys, json, time, os

from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

fr_UTC = '20150101 00:00:00'
to_UTC = '20170101 00:00:00'
fr_utc = int(1e9 * int(time.strftime('%s', time.strptime(fr_UTC, '%Y%m%d %H:%M:%S'))))
to_utc = int(1e9 * int(time.strftime('%s', time.strptime(to_UTC, '%Y%m%d %H:%M:%S'))))

# for online access
ArchiveDB = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/ProjectDesc.1/ProgramLabelLog/parms/{key}/_signal.json?from={fr_utc}&upto={to_utc}'
#ArchiveDB = 'file:///data/databases/W7X/cache/{key}.json'  # for local files

shotinfo = {}
for itm in ['comment','ProgramLogLabel/progId','ProgramLogLabel/date']:
    key = itm.split('/')[-1]  # choose the last part of the name
    url = ArchiveDB.format(key=itm,fr_utc=fr_utc,to_utc=to_utc)
    shotinfo.update({key: json.load(urlopen(url))})

shotDA = {}
for k in shotinfo:
    shotDA.update({k: shotinfo[k]['values'][::2]})

shotDA.update({'start_utc': shotinfo['date']['dimensions'][::2]})
shotDA.update({'end_utc': shotinfo['date']['dimensions'][1::2]})

for k in ['date','progId', 'start_utc', 'end_utc']:
  shotDA.update({k: np.array(shotDA[k], np.int)})  # make it an array of ints

d = np.unique(shotDA['date'])
wL = np.where((shotDA['end_utc']-shotDA['start_utc'])>60e9)[0]
print('Over {d} days, there were {n} shots, {N} longer than 100ms'
      .format(N=len(wL), d=len(d), n=len(shotDA['date'])))

#for i in range(10): 
for i in range(-110,-1): 
    print(shotDA['date'][i],shotDA['progId'][i],time.ctime(shotDA['end_utc'][i]/1e9),
          1e-9*(shotDA['end_utc'][i]-shotDA['start_utc'][i]),shotDA['comment'][i])

import pickle
this_dir = os.path.dirname(__file__)
pickle.dump(shotDA, open(os.path.join(this_dir,'shotDA.pickle'),'w'))
