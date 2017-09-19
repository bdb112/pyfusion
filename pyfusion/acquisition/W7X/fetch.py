""" W7-X data fetchers. 

Since December 8th 2016, cached npz files include the raw utcs (differenced) at very little additional cost
in space.  - see examples/read_W7X_json.py

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

import sys
if sys.version < '3.0.0':
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
from pyfusion.data.base import Coords, Channel, ChannelList, \
    get_coords_for_channel

import json

from pyfusion.debug_ import debug_
import sys, os
import time as tm

from .get_shot_info import get_shot_utc

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
    msg = None  # msg allows us to see which shot/diag was at fault
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
    histo = plt.hist if pyfusion.DBG() > 1 else np.histogram
    cnts, vals = histo(x, bins=200)[0:2]
    # ignore the two end bins - hopefully there will be very few there
    wmin = np.where(cnts[1:-1] < np.max(cnts[1:-1]))[0]
    if len(wmin)>0:
        print('**********\n*********** Gap in data > {p:.2f}%'.format(p=100*len(wmin)/float(len(cnts))))
    x01111 = np.ones(len(x))  # x01111 will be all 1s except for the first elt.
    x01111[0] = 0
    errcnt = np.sum(bigcounts) + np.sum(np.sort(counts)[::-1][1:])
    if errcnt>0 or (pyfusion.VERBOSE > 0): 
        msg = str('** repaired length of {l:,}, dtns={dtns:,}, {e} erroneous utcs'
              .format(l=len(x01111), dtns=dtns, e=errcnt))

    fixedx = np.cumsum(x01111)*dtns
    wbad = np.where((x - fixedx)>1e8)[0]
    fixedx[wbad] = np.nan
    debug_(pyfusion.DEBUG, 3, key="repair", msg="repair of W7-X scrambled Langmuir timebase") 
    return(fixedx, msg)
    

class W7XDataFetcher(BaseDataFetcher):
    """Fetch the W7X data using urls"""

    def do_fetch(self):
        # my W7X shots are of the form from_utc, to_utc 
        #  or date (8dig) and shot (progId)
        # the format is in the acquisition properties, to avoid
        # repetition in each individual diagnostic

        if self.shot[1]>1e9:  # we have start and end in UTC
            f,t = self.shot
        else:
            f,t = get_shot_utc(self.shot)
        # A URL STYLE diagnostic - used for a one-off
        # this could be moved to setup so that the error info is more complete
        if hasattr(self,'url'):
            fmt = self.url+'_signal.json?from={shot_f}&upto={shot_t}'
            params = {}
        else:  # a pattern-based one - used for arrays of probes
            if hasattr(self,'fmt'):  #  does the diagnostic have one?
                fmt = self.fmt
            elif hasattr(self.acq,'fmt'):  # else use the acq.fmt
                fmt = self.acq.fmt
            else:  #  so far we have no quick way to check the server is online
                raise LookupError('no fmt - perhaps pyfusion.cfg has been '
                                  'edited because the url is not available')

            params = eval('dict('+self.params+')')

        if 'upto' not in fmt:
            fmt += '_signal.json?from={shot_f}&upto={shot_t}'

        # nSamples now needs a reduction mechanism http://archive-webapi.ipp-hgw.mpg.de/
        # minmax is increasingly slow for nSamples>10k, 100k hopeless
        # should ignore the test comparing the first tow elements of the tb
        if ('nSamples' not in fmt) and (pyfusion.NSAMPLES != 0):
            fmt += '&reduction=minmax&nSamples={ns}'.format(ns=pyfusion.NSAMPLES)

        params.update(shot_f=f, shot_t=t)
        url = fmt.format(**params)
        # fix up erroneous ECH alias mapping if ECH - only 6 work if I don't
        #  Hopefully at some point in the future, they will fix it.
        # this implementation is kludgey but proves the principle
        if 'Rf' in url or 'MainCoils' in url:
            from pyfusion.acquisition.W7X.get_url_parms import get_signal_url
            # replace the main middle bit with the expanded one from the GUI
            tgt = url.split('KKS/')[1].split('/scaled')[0]
            url = url.replace(tgt, get_signal_url(tgt)).split('KKS/')[-1]

        debug_(pyfusion.DEBUG, 3, key="url", msg="work on urls")
        # we need %% in pyfusion.cfg to keep py3 happy
        # however with the new get_signal_url, this will all disappear
        if sys.version < '3.0.0' and '%%' in url:
            url = url.replace('%%','%')

        if 'StationDesc.82' in url:  # fix spike bug in scaled QRP data
            url = url.replace('/scaled/', '/unscaled/')
            
        if pyfusion.CACHE:
            # needed for improperly configured cygwin systems: e.g.IPP Virual PC
            # perhaps this should be executed at startup of pyfusion?
            cygbin = "c:\\cygwin\\bin"
            if os.path.exists(cygbin) and not cygbin in os.environ['path']:
                os.environ['path'] += ";" + cygbin
            print('using wget on {url}'.format(url=url))
            retcode = os.system('wget -x "{url}"'.format(url=url))
            #  retcode = os.system('c:\\cygwin\\bin\\bash.exe -c "/bin/wget {url}"'.format(url=url))
            debug_(retcode != 0, level=1, key='wget', msg="wget error or DEBUG='wget'")
            # now read from the local copy - seems like urls need full paths
            # appears to be a feature! http://stackoverflow.com/questions/7857416/file-uri-scheme-and-relative-files
            # /home/bdb112/pyfusion/working/pyfusion/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.181_DATASTREAM/7/Channel_7/scaled/_signal.json?from=1457626020000000000&upto=1457626080000000000&nSamples=10000
            # url = url.replace('http://','file:///home/bdb112/pyfusion/working/pyfusion/')
            url = url.replace('http://','file:/'+os.getcwd()+'/')
            if 'win' in os.sys.platform:
                # weven thoug it seems odd, want 'file:/c:\\cygwin\\home\\bobl\\pyfusion\\working\\pyfusion/archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.82/DataModuleDesc.192_DATASTREAM/4/Channel_4/scaled/_signal.json@from=147516863807215960&upto=1457516863809815961'
                url= url.replace('?','@')
            # nicer replace - readback still fails in Win, untested on unix systems
            print('now trying the cached copy we just grabbed: {url}'.format(url=url))
        if pyfusion.VERBOSE > 0:
            print('======== fetching url over {dt:.1f} secs  =========\n[{u}]'
                  .format(u=url, dt=(params['shot_t'] - params['shot_f'])/1e9))

        # seems to take twice as long as timeout requested.
        # haven't thought about python3 for the json stuff yet
        try:
            # dat = json.load(urlopen(url,timeout=pyfusion.TIMEOUT)) works
            # but follow example in
            #    http://webservices.ipp-hgw.mpg.de/docs/howtoREST.html#python, 
            dat = json.loads(urlopen(url,timeout=pyfusion.TIMEOUT).read().decode('utf-8'))
        except socket.timeout:
            # should check if this is better tested by the URL module
            print('****** first timeout error *****')
            dat = json.load(urlopen(url,timeout=3*pyfusion.TIMEOUT))
        except Exception as reason:
            if pyfusion.VERBOSE:
                print('********Exception***** on {c}: {u} \n{r}'
                      .format(c=self.config_name, u=url, r=reason))
            raise

        # this form will default to repair = 2 for all LP probes.
        default_repair = 2 if 'Desc.82/' in url else 0
        # this form follows the config file settings
        self.repair = int(self.repair) if hasattr(self, 'repair') else default_repair
        dimraw = np.array(dat['dimensions'])  
        dim = dimraw - dimraw[0]
        if self.repair == 0:
            pass # leave as is
        # need at least this clipping for Langmuir probes in Op1.1
        elif self.repair == 1:
            dim = np.clip(dim, 0, 1e99)
        elif self.repair == 2:
            dim, msg = regenerate_dim(dim)
            if msg is not None:
                print('shot {s}, {c}: {m}'
                      .format(s=self.shot, c=self.config_name, m=msg))
        else:
            raise ValueError('repair value of {r} not understood'.format(r=self.repair))

        if pyfusion.VERBOSE>2:  print('repair',self.repair)
        #ch = Channel(self.config_name,  Coords('dummy', (0,0,0)))
        # this probably should be in base.py
        coords = get_coords_for_channel(**self.__dict__)
        # used to be bare_chan? should we include - signs?
        ch = Channel(self.config_name,  coords)
        scl = 1/3277.24 if dat['datatype'].lower() == 'short' else 1
        output_data = TimeseriesData(timebase=Timebase(1e-9*dim),
                                     signal = scl*Signal(dat['values']), channels=ch)
        output_data.meta.update({'shot': self.shot})
        output_data.utc = [dat['dimensions'][0], dat['dimensions'][-1]]
        output_data.units = dat['units'] if 'units' in dat else ''
        # this is a minor duplication - at least it gets saved via params
        params['data_utc'] = output_data.utc
        # Warning - this could slow things down! - but allows 
        # corrupted time to be re-calculated as algorithms improve.
        # and storage as differences takes very little space.
        params['diff_dimraw'] = dimraw
        params['diff_dimraw'][1:] = np.diff(dimraw)
        # NOTE!!! need float128 to process dimraw, and cumsum won't return ints
        # or automatically promote to 128bit (neither do simple ops like *, +)
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

        # A kludge to warn user if there is no access to the URL data
        # this can be used by H-1 URL also, if so, needs to be at a higher level
        # In the current nameserver list, presence of sv indicates IPP internal
        #  very dodgy test - any better ideas?
        if ((pyfusion.LAST_DNS_TEST < 0) 
            or ((tm.time() - pyfusion.LAST_DNS_TEST < 10)
                and (hasattr(self.acq, 'access')))):
            if pyfusion.VERBOSE>0:
                print('Skipping URL access check - see pyfusion.LAST_DNS_TEST')
            return
        # no else - the return (above) does the job

        pyfusion.LAST_DNS_TEST = tm.time()
        if not hasattr(self.acq, 'access'):
            self.acq.access = False

        dominfo = dict(dom=self.acq.domain, look=self.acq.lookfor, 
                       access=self.acq.access)

        self.acq.access = False
        if pyfusion.VERBOSE>0:
            print('Doing URL access check - see pyfusion.LAST_DNS_TEST')

        try:
            # this is the clean way
            import dns.resolver
            domain = self.acq.domain
            answers = dns.resolver.query(domain,'NS')
            for (iii, server) in enumerate(answers):
                if self.acq.lookfor in server.to_text():
                    self.acq.access = True 
                if pyfusion.VERBOSE>0: 
                    dominfo.update(dict(access=self.acq.access))
                    print(server.to_text())
                    print('dns.resolver test for {look} in nameserver {dom} indicates access is {access}'
                          .format(**dominfo))
        except ImportError:
            # this alternative may work, especially if cygwin is installed, but is noisier
            # note - not set up for cygwin yet - need to call bash so we have access to grep
            try:
                #os.system('nslookup '+self.acq.domain)
                self.acq.access = (
                    # should really check for 0 (found) 1Win/256Linux (not found) 
                    #  or 256W >>256 Lin  (error)
                    (0 == os.system('nslookup {dom}|find "{look}" 2> /tmp/winjunk'.format(**dominfo))) or
                    (0 == os.system('nslookup {dom}|grep {look} 2> /tmp/linjunk'.format(**dominfo))))

                if pyfusion.VERBOSE>0: 
                    print('nslookup test looking for {look} on nameserver {dom} indicates access is {access}'
                          .format(**dominfo))
            except ():
                print('*******************************************************************************')
                print('Warning- assuming we have archive network access in the absence of any real information')
                print('*******************************************************************************')
                self.acq.access = None

        if self.acq.access is False:
            raise LookupError('URL for {d} in {s} is probably not accessible or the\n '\
                              'necessary software (dnspython or nslookup/grep) is not installed.\n'\
                              'Set pyfusion.LAST_DNS_TEST=-1 to skip test'
                              .format(d=self.config_name, s=self.shot))
