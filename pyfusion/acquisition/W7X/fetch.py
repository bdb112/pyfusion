""" W7-X data fetchers. 
Example of data reaching over a break in time
run pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name=W7X_L53_LP1 shot_number=[1457015819400000000,1457015821000000000] hold=1 sharey=2
best to limit nSamples to ~ 10k - see pyfusion.cfg

Test method: Clumsy
uncomment file:// in pyfusion.cfg
from pyfusion.acquisition.W7X import fetch
ff=fetch.W7XDataFetcher("W7X_L53_LP7_I",[20160302,26])
ff.fmt='file:///data/databases/W7X/cache/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.{CDS}/DataModuleDesc.{DMD}_DATASTREAM/{ch}/Channel_{ch}/scaled/_signal.json?from={shot_f}&upto={shot_t}'
ff.params='CDS=82,DMD=190,ch=2'
ff.do_fetch()
ff.repair=1   # or 0,2

Would be better to have an example with non-trivial units
"""

from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import socket

import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pyfusion
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, ChannelList, Channel
import json

from pyfusion.debug_ import debug_
import sys

from .get_shot_utc import get_shot_utc

VERBOSE = pyfusion.VERBOSE


def myhist(X):
    """ use a dict method to do a bincount on large or negative numbers 
    Unwise to use this alone on W7-X because there are easily 1e9 instances 
    We use bincount for the smaller numbers"""
    his = {}
    for x in X:
        if x in his:
            his[x] += 1
        else:
            his[x] = 1

    counts, vals = [], []
    for (i, x) in enumerate(his):
        counts.append(his[x])
        vals.append(x)
    return(counts, vals)


def regenerate_dim(x):
    """ assume x in ns since epoch from the current time """
    diffs = np.diff(x)
    # bincount needs a positive input and needs an array with N elts where N is the largest number input
    small = (diffs > 0) & (diffs < 1000000)
    sorted_diffs = np.sort(diffs[np.where(small)[0]])
    counts = np.bincount(sorted_diffs)
    bigcounts, bigvals = myhist(diffs[np.where(~small)[0]])

    if pyfusion.VERBOSE>0:
        print('[[diff, count],....]')
        print('small:', [[argc, counts[argc]] for argc in np.argsort(counts)[::-1][0:5]])
        print('big or negative:', [[bigvals[argc], bigcounts[argc]] for argc in np.argsort(bigcounts)[::-1][0:10]])

    dtns = 1 + np.argmax(counts[1:])  # skip the first position - it is 0
    # wgt0 = np.where(sorted_diffs > 0)[0]  # we are in ns, so no worry about rounding
    histo = plt.hist if pyfusion.DEBUG>1 else np.histogram
    cnts, vals = histo(x, bins=200)[0:2]
    # ignore the two end bins - hopefully there will be very few there
    wmin = np.where(cnts[1:-1] < np.max(cnts[1:-1]))[0]
    if len(wmin)>0:
        print('**********\n*********** Gap in data > {p:.2f}%'.format(p=100*len(wmin)/float(len(cnts))))
    x01111 = np.ones(len(x))  # x01111 will be all 1s except for the first elt.
    x01111[0] = 0
    errcnt = np.sum(bigcounts) + np.sum(np.sort(counts)[::-1][1:])
    if errcnt>0 or (pyfusion.VERBOSE > 0): 
        print('** repaired length of {l:,}, dtns={dtns:,}, {e} erroneous utcs'
              .format(l=len(x01111), dtns=dtns, e=errcnt))

    fixedx = np.cumsum(x01111)*dtns
    wbad = np.where((x - fixedx)>1e8)[0]
    fixedx[wbad] = np.nan
    debug_(pyfusion.DEBUG, 3, key="repair", msg="repair of W7-X scrambled Langmuir timebase") 
    return(fixedx)
    

class W7XDataFetcher(BaseDataFetcher):
    """Fetch the W7X data using urls"""

    def do_fetch(self):
        # my W7X shots are of the form from_utc, to_utc 
        #  or date (8dig) and shot (progId)
        # the format is in the acquisition properties, to avoid
        # repetition in each individual diagnostic
        ch = Channel(self.config_name,  Coords('dummy', (0,0,0)))
        if self.shot[1]>1e9:  # we have start and end in UTC
            f,t = self.shot
        else:
            f,t = get_shot_utc(*self.shot)
        # A URL STYLE diagnostic - used for a one-off
        if hasattr(self,'url'):
            fmt = self.url+'_signal.json?from={shot_f}&upto={shot_t}'
            fmt = self.url+'_signal.json?from={shot_f}&upto={shot_t}&nSamples=200000'
            params = {}
        else:  # a pattern-based one - used for arrays of probes
            if hasattr(self,'fmt'):
                fmt = self.fmt
            else:
                fmt = self.acq.fmt
            params = eval('dict('+self.params+')')

        if 'upto' not in fmt:
            fmt += '_signal.json?from={shot_f}&upto={shot_t}'

        if ('nSamples' not in fmt) and (pyfusion.NSAMPLES != 0):
            fmt += '&nSamples={ns}'.format(ns=pyfusion.NSAMPLES)

        params.update(shot_f=f, shot_t=t)
        url = fmt.format(**params)
        if pyfusion.VERBOSE > 0:
            print('===> fetching url {u}'.format(u=url))

        # seems to take twice as long as timeout requested.
        # haven't thought about python3 for the json stuff yet
        try:
            dat = json.load(urlopen(url,timeout=30))
        except socket.timeout:
            # should check if this is better tested by the URL module
            print('****** first timeout error *****')
            dat = json.load(urlopen(url,timeout=60))
        except Exception as reason:
            print('********Exception***** on {c}: {u} \n{r}'
                  .format(c=self.config_name, u=url, r=reason))
            raise

        # this form will default to repair = 2 for all LP probes.
        default_repair = 2 if 'Desc.82/' in url else 0
        # this form follows the config file settings
        self.repair = int(self.repair) if hasattr(self, 'repair') else default_repair
        dim = np.array(dat['dimensions']) - dat['dimensions'][0]
        if self.repair == 0:
            pass # leave as is
        # need at least this clipping for Langmuir probes in Op1.1
        elif self.repair == 1:
            dim = np.clip(dim, 0, 1e99)
        elif self.repair == 2:
            dim = regenerate_dim(dim)
        else:
            raise ValueError('repair value of {r} not understood'.format(r=self.repair))

        if pyfusion.VERBOSE>2:  print('repair',self.repair)
        output_data = TimeseriesData(timebase=Timebase(1e-9*dim),
                                     signal=Signal(dat['values']), channels=ch)
        output_data.meta.update({'shot': self.shot})
        output_data.utc = [dat['dimensions'][0], dat['dimensions'][-1]]
        output_data.units = dat['units'] if 'units' in dat else ''
        # this is a minor duplication - at least it gets saved via params
        params['data_utc'] = output_data.utc
        params['pyfusion_version'] = pyfusion.version.get_version()
        if pyfusion.VERBOSE > 0:
            print('shot {s}, config name {c}'
                  .format(c=self.config_name, s=self.shot))

        output_data.config_name = self.config_name
        debug_(pyfusion.DEBUG, 2, key='W7XDataFetcher')
        output_data.params = params
        
        ###  the total shot utc.  output_data.utc = [f, t]
        return output_data

    def setup(self):
        """ record some details of the forthcoming fetch so 
        that the calling routine in base can give useful error messages
        The is info may not be available in full if the fetch fails.
        This was not the original purpose of setup, it was more
        general than this. (as part of a 3 step process:
        setup, do_fetch, pulldown
        """
        self.fetch_mode = 'http'
                    
