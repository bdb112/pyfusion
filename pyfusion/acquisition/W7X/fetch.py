""" W7-X data fetchers. 
Example of data reaching over a break in time
run pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name=W7X_L53_LP1 shot_number=[1457015819400000000,1457015821000000000] hold=1 sharey=2
best to limit nSamples to ~ 10k - see pyfusion.cfg
"""

from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

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

def regenerate_dim(x):
    """ assume x in ns since epoch from the current time """
    diffs = np.diff(x)
    sorted_diffs = np.sort(diffs[np.where((diffs > 0) & (diffs < 100000))[0]])
    counts = np.bincount(sorted_diffs)
    if pyfusion.VERBOSE>1:
        print([[argc, counts[argc]] for argc in np.argsort(counts)[::-1][0:3]])
    dtns = 1 + np.argmax(counts[1:])  # skip the first position - it is 0
    # wgt0 = np.where(sorted_diffs > 0)[0]  # we are in ns, so no worry about rounding
    histo = plt.hist if pyfusion.DEBUG>1 else np.histogram
    cnts, vals = histo(x, bins=200)[0:2]
    # ignore the two end bins - hopefully there will be very few there
    wmin = np.where(cnts[1:-1] < np.max(cnts[1:-1]))[0]
    if len(wmin)>0:
        print('Gap in data > {p:.2f}%'.format(p=100*len(wmin)/float(len(cnts))))
    x01111 = np.ones(len(x))  # x01111 will be all 1s except for the first elt.
    x01111[0] = 0
    if pyfusion.VERBOSE>0: 
        print('********** repaired length {l:,}, dtns={dtns:,}'
              .format(l=len(x01111), dtns=dtns))
    fixedx = np.cumsum(x01111)*dtns
    wbad = np.where((x - fixedx)>1e8)[0]
    fixedx[wbad] = np.nan
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
            fmt = self.url+'_signal.json?from={f}&upto={t}'
            fmt = self.url+'_signal.json?from={f}&upto={t}&nSamples=200000'
            params = {}
        else:  # a pattern-based one - used for arrays of probes
            fmt = self.acq.fmt
            params = eval('dict('+self.params+')')

        params.update(f=f, t=t)
        url = fmt.format(**params)
        if pyfusion.VERBOSE > 0:
            print('===> fetching url {u}'.format(u=url))

        # seems to take twice as long as timeout requested.
        # haven't thought about python3 for the json stuff yet
        try:
            dat = json.load(urlopen(url,timeout=20))
        except URLError as reason:
            print('timeout on {c}: {u} \n{r}'
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

        output_data = TimeseriesData(timebase=Timebase(1e-9*dim),
                                     signal=Signal(dat['values']), channels=ch)
        output_data.meta.update({'shot': self.shot})
        output_data.utc = [dat['dimensions'][0], dat['dimensions'][-1]]

        if pyfusion.VERBOSE > 0:
            print('shot {s}, config name {c}'
                  .format(c=self.config_name, s=self.shot))

        output_data.config_name = self.config_name

        debug_(pyfusion.DEBUG, 2, key='W7XDataFetcher')
        # output_data.params = stprms
        output_data.utc = [f, t]
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
                    
