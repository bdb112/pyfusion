""" Obtain a dictionary containing all the TDU and Striker Plate Langmuir probe data

Useful lines
pyfusion.W7X_minerva_cache.items()[0][1]['parms']['validSince']['dimensions'][0]  1504788000000000000
pyfusion.W7X_minerva_cache.items()[0][1]['parms']['modifiedAt']['dimensions'][0]] 1504788000000000000



"""
from __future__ import print_function
from future.standard_library import install_aliases
from collections import OrderedDict
import json
import sys, glob
import numpy as np
if sys.version < '3.0.0':         # 'fixups' for py2 compat.
    FileNotFoundError = OSError   #  OSError is preferred over IOError in stackX

import time, calendar

import pyfusion
from pyfusion.acquisition.H1.scrape_wiki import get_links
from pyfusion.debug_ import debug_
from get_shot_info  import get_shot_utc
from pyfusion.utils.time_utils import utc_ns
from pyfusion.data.shot_range import shot_gte

install_aliases()
from urllib.request import urlopen, Request
#  https://codereview.stackexchange.com/questions/21033/flatten-dictionary-in-python-functional-style

def flatten_dict(d, level=None, sep='.'):
    """ return a dictionary flatten to a  depth of <level> 
    level = 0 is a noop
    level = None returns a totally flattened dictionary
    """
    def items():
        # somehow this function prevents a proper traceback - so I put a check a few lines below.
        for key, value in d.items():
            if (level is None or level>0) and isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value, level=level-1 if level is not None else None, sep=sep).items():
                    yield key + sep + subkey, subvalue
            else:
                yield key, value

    if d is None:
        raise ValueError('dictionary to be flattened is empty')
    return dict(items())

# This (YvesgereY stack overflow version separates out the key concatenation - might be more flexible)
def iteritems_nested(d):
  def fetch (suffixes, v0) :
    if isinstance(v0, dict):
      for k, v in v0.items() :
        for i in fetch(suffixes + [k], v):  # "yield from" in python3.3
          yield i
    else:
      yield (suffixes, v0)

  return fetch([], d)

def flatten_dict_by_concatenating_keys(d) :
  return dict( ('.'.join(ks), v) for ks, v in iteritems_nested(d))
  #return { '.'.join(ks) : v for ks,v in iteritems_nested(d) }
  

class MinervaMap(object):
    """ read a minerva map into a nested dictionary.  Shot can be a shot or a utc_ns
    """
    def __init__(self, shot):
        # fname =  "http-__archive-webapi.ipp-hgw.mpg.de_ArchiveDB_raw_Minerva_Minerva.QRP.settings_Settings_PARLOG_V25_.json"
        # special case when shot is a single utc_ns
        if (len(np.shape(shot)) == 0) and shot > 1e9:
            shot = [shot, shot]

        parm_URL = get_suitable_version(shot)
        if parm_URL is None:
            raise LookupError('No Minerva mapping for ', shot)
        fname = sanitised_json_filename(parm_URL)
        self.parm_URL = parm_URL

        try:
            self.parmdict = json.load(open(fname, 'rt'))
        except FileNotFoundError:
            print('need to run pyfusion/acquisition/W7X/get_url_parms.py  # to generate minerva cache')
            
    def get_parm(self, pseudourl=None):
        return get_parm(pseudourl, self.parmdict)
                
    
def get_parm(pseudourl, parmdict, default=None):
    """ Example: to get probe resistance
    get_parm('parms/probes/upperTestDivertorUnit/Probe_2/adc1/channels/channel2/modeResistor/values', parmdict)[0]
    if the path is not valid or doesn't exist, return None
    """
    if pseudourl.endswith('/'):
        pseudourl = pseudourl[0:-1]

    subdict = parmdict.copy()
    for key in pseudourl.split('/'):
        if not key in subdict:
            if pyfusion.VERBOSE>0:
                print('*** url {pse} not found in chan_dict'.format(pse=pseudourl))
            return default
        subdict = subdict[key]

    return(subdict)



def get_minerva_parms(fetcher_obj):
    if shot_gte([20170101,0], fetcher_obj.shot):
        print('Warning - looking up a pre-minerva shot { shot} '.format(shot=fetcher_obj.shot))
        # return(fetcher_obj, '')

    try:
        mm = MinervaMap(fetcher_obj.shot)
    except LookupError as reason:
        raise LookupError('No Minerva mapping for {cn} as {mn} for shot {sh} {reason}'
                          .format(mn=fetcher_obj.minerva_name, cn=fetcher_obj.config_name,
                                  sh=fetcher_obj.shot, reason = str(reason)))
    
    last_mod = mm.get_parm('parms/modifiedAt/values')[0]
    cal_remarks = mm.get_parm('parms/generalRemarks/values')[0] 
    print('last mod at ', last_mod, cal_remarks, end='\t')
    # go through all the channels mentioned, check mode according to I or V type
    chan_dict = mm.get_parm('parms/probes/{mn}'.format(mn=fetcher_obj.minerva_name))
    if chan_dict is None:
        raise LookupError('\n Minerva name {mn} not found in {mmURL}'
                          .format(mn=fetcher_obj.minerva_name, mmURL=mm.parm_URL))
    chans=[ch for ch in flatten_dict(chan_dict, 2,sep='/').keys() if 'channels' in ch]
    if fetcher_obj.config_name.endswith('I'):
        chans =[ch for ch,chall in
                [(chn, flatten_dict(get_parm(chn, chan_dict)))
                 for chn in chans] if np.any(['istor.v' in k for k in chall])]
    elif fetcher_obj.config_name.endswith('U'):
        chans =[ch for ch,chall in
                [(chn, flatten_dict(get_parm(chn, chan_dict)))
                 for chn in chans] if np.any(['actor.v' in k for k in chall])]
    if len(chans) == 0:
        raise LookupError('No suitable monitoring chans found for {mn} on {shot}'
                          .format(mn=fetcher_obj.minerva_name, shot=fetcher_obj.shot))
    elif len(chans) > 1:
        debug_(pyfusion.DEBUG, 2, key='MinervaName', msg='leaving MinervaName')
        if fetcher_obj.config_name.endswith('U'):
            print('Warning - more than one channel{ch} - will allow for voltages (sweep)'
                  .format(ch=fetcher_obj.config_name))
            chan = chans[0]
        else:
            raise ValueError('Too many chans found for {mn} on {shot}'
                             .format(mn=fetcher_obj.minerva_name, shot=fetcher_obj.shot))

    else:  # everything is fine, there is only one match.
        chan = chans[0]

    electronics = get_parm(chan, chan_dict)
    adc, junk, channel = chan.split('/')
    chan_no = int(channel[7:]) - 1
    adc_no = int(adc[3:])
    # gain and params are in string representation in the OP1.1, so do the same here
    rs_used = None  #  later we will save rs_used whether I or V
    if fetcher_obj.config_name.endswith('I'):
        if 'modeResistor' not in electronics:
            raise LookupError('current not recorded for {mn} on {shot}, adc {adc}'
                              .format(mn=fetcher_obj.minerva_name, shot=fetcher_obj.shot, adc=adc))

        rs = float(electronics['modeResistor']['values'][0]) # V38 up has strings here
        rs_used = rs*1.0  # save rs_used in npz so we can track correction method
        if utc_ns(last_mod) < utc_ns('20171130'):
            pyfusion.utils.warn('fudgeing gain correction until resistances corrected')
            rs_used += 1. # original very rough way to allow for cross talk effect
        else:
            pyfusion.utils.warn('assuming no need to fudge rs')
        gainstr = str(1/(rs_used*electronics['driverElectronicTotalGain']['values'][0])) + ' A' 

    else:
        modeFactor = float(electronics['modeFactor']['values'][0])
        rs = None
        gainstr = str(1/(modeFactor*electronics['driverElectronicTotalGain']['values'][0])) + ' V'

    if hasattr(fetcher_obj, 'gain'):
        pyfusion.utils.warn('Overriding gain {g} from config file in {nm} - conflict??'
                            .format(nm=fetcher_obj.config_name, g=str(fetcher_obj.gain)))
    fetcher_obj.gain = gainstr
    CDS = 82 # until Op1.2b
    dmd = 180 + adc_no  # adc1 is the left most, and maps to 181_DATASTREAM
    if adc_no > 4:
        dmd += 4
        if shot_gte(fetcher_obj.shot, [20180101,0]):
            CDS = 26147

    fetcher_obj.params = str("CDS={CDS},DMD={dmd},ch={ch},rs={rs},rs_used={rs_used}"
                             .format(CDS=CDS, dmd=dmd, ch=chan_no, rs=rs,rs_used=rs_used))
    PS = get_parm(chan+'/powerSupply/values', chan_dict)
    if PS is not None:
        fetcher_obj.params += str(',powerSupply="{PS}"'.format(PS=PS[0]))
    debug_(pyfusion.DEBUG, 1, key='MinervaName', msg='leaving MinervaName')
    print('rs_used={rs_used}, overall gain={gain}, params={params}\t'
          .format(rs_used=rs_used, gain=fetcher_obj.gain, params=fetcher_obj.params))
    return(fetcher_obj, dict(cal_date=last_mod, cal_comment=cal_remarks))


def get_subtree(url, debug=1, level=0):
    """  only works for urls
               level is just for pretty printing 
    """
    dic = OrderedDict()   # Don't forget the parens, Boyd
    links = get_links(url, debug=0)
    for lnk, name in links:
        name = name.strip()
        if url not in lnk or 'Remove' in name:
            continue
        if lnk.endswith('/'):
            if pyfusion.VERBOSE > 0: print('recursing into ', level*'    ', lnk.replace(url,'-'), name)
            if name in dic:
                raise LookupError(' dictionary already contains ' + name)
            else:
                dic.update({name: get_subtree(lnk, debug=debug, level=level + 1)})
        else:
            if debug > 1: print('retrieving data from ', name)
            daturl = lnk + '/_signal.json?from=0&upto=1606174400000000000'
            dat = json.loads(urlopen(daturl, timeout=pyfusion.TIMEOUT).read().decode('utf-8'))
            if name in dic:
                raise LookupError(' dictionary already contains ' + name)
            else:
                dic.update({name: dat})
    return(dic)

def get_signal_url(path='CBG_ECRH/A1/medium_resolution/Rf_A1', filter=''):
    """ return the cryptic url corresponding to the human readable form relative to 
    ArchiveDB/views/KKS
    """
    import os
    root = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/views/KKS/'
    oneup, wantname = os.path.split(path)
    fullurl = os.path.join(root, oneup)
    if len(filter) > 0:
        fullurl += '/' + filter
    links = get_links(fullurl, debug=0, exceptions=Exception)
    if links is None:
        if "filterstart" in fullurl:
            raise LookupError('Page not found in that time range', fullurl)
        else:
            raise LookupError("Page not found", fullurl)
    debug_(pyfusion.DEBUG, 1, key="signal_url", msg="check url link list")
    for lnk, name in links:
        name = name.strip()
        if 'DATASTREAM' not in lnk or 'Remove' in name:
            continue
        if name == wantname:
            print('Found!')
            debug_(pyfusion.DEBUG, 1, key="after_signal_url", msg="check url link list")
            return(lnk)
    debug_(pyfusion.DEBUG, 1, key="after_signal_url", msg="check url link list")
    print('>>>  ', wantname +' not in ', np.sort([n.strip() for l,n in links]))
    raise LookupError(path + ' not found under ' + root)


def get_PARMLOG_cache(shot):
    """ this is very similar to get_suitable params, but uses a local file cache
    The cache should be persistent across 'run' commands - an easy way
    is to initialise in pyfusion.
 """
    Vfiles = glob.glob("*PARLOG_V[0-9]*json")
    # put them in a dictionary as a cache - should live in an object
    # (maybe MinervaMap, but save in pyfusion,,,, for now)
    if not hasattr(pyfusion, 'W7X_minerva_cache') or pyfusion.W7X_minerva_cache is None:
        print('>>>>>> loading pyfusion.W7X_minerva_cache', end='..')
        pyfusion.W7X_minerva_cache = dict([[fn, json.load(open(fn, 'rt'))] for fn in Vfiles])

    vnumstrs = [vfile.split('PARLOG_V')[-1].split('_.json')[0] for vfile in Vfiles]
    vnums  = np.array([int(vn) for vn in vnumstrs if vn is not None])
    sortd = np.argsort(vnums)[::-1]
    utcs = get_shot_utc(shot)
    # now work down thourgh the versions and find the first where shot utc is after valid
    for vkey in np.array(Vfiles)[sortd]:
        debug_(pyfusion.DEBUG, 1, key='get_PARMLOG_cache', msg='reading PARLOG_V files')
        if get_parm('parms/validSince/dimensions',pyfusion.W7X_minerva_cache[vkey])[0] < utcs[0]:
            pyfusion.utils.warn('Changing to parmlog ' + vkey)
            return(vkey)
    return(None)    

    
def get_suitable_version(shot=[20170921,10], debug=1):
    local_copy = get_PARMLOG_cache(shot)
    if local_copy is not None:
        return(local_copy)
    # effectively an else...
    utcs = get_shot_utc(shot)
    minerva_parms_url = "http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/views/Minerva/Diagnostics/LangmuirProbes/Settings/"
    links = get_links(minerva_parms_url, debug=debug)
    url_dict = dict([[lnk[1].strip(), lnk[0]] for lnk in links if lnk[1].strip().startswith('V')])
    desc_order = np.argsort([int(version[1:]) for version in list(url_dict)])[::-1]
    vers_desc = np.array(list(url_dict))[desc_order]
    # Work down through the names until we find a file whose valid date predates the shot
    # There is a way to define the end of validity other than implicit overriding by a
    #   higher version number  but I don't know
    # Do with a for loop to save looking unnecessarily deep
    for ver in vers_desc:
        val_url = url_dict[ver] + 'parms/validSince/'
        daturl = val_url + '/_signal.json?from=0&upto=1606174400000000000'
        dat = json.loads(urlopen(daturl, timeout=pyfusion.TIMEOUT).read().decode('utf-8'))
        if debug>0:
            print(int(1e9) * int(calendar.timegm(time.strptime(dat['values'][0], '%Y-%m-%d %H:%M:%S:%f'))),
                  dat['dimensions'])
        if int(dat['dimensions'][0]) < utcs[0]:
            return(url_dict[ver])
    return(None)

def sanitised_json_filename(URL):
    if (not URL.endswith('.json')) and (not URL.endswith('/')):
        URL += '/'
    fname = URL.replace('/','_').replace(':','-')
    if not fname.lower().endswith('.json'):
         fname += '.json'
    return(fname)


if __name__ == '__main__':
    URL = sys.argv[1] if len(sys.argv) > 1 else 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/Minerva/Minerva.QRP.settings/Settings_PARLOG/V27/'
    get_PARMLOG_cache([20170927,10])

    parmdict = get_subtree(URL)
    if 'parms' not in parmdict:
        raise LookupError("Probably couldn't find {URL}".format(URL=URL))
    
    if len(list(parmdict['parms'])) < 5:
        print('Warning - parmdict is too short')
    if len(list(parmdict)) == 0:
        raise LookupError('No Minerva data parameters found in \n' + URL)
    fname = sanitised_json_filename(URL)
    print('saving in ', fname)
    json.dump(parmdict, open(fname, 'wt'))


"""
#  Basis of method - examples

import pyfusion
import json
from pyfusion.acquisition.H1.scrape_wiki import get_links
links = get_links('http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/Minerva/Minerva.QRP.settings/Settings_PARLOG')

# Then to extract a value, choose the latest version (how?)
from future.standard_library import install_aliases
install_aliases()
from urllib.request import urlopen, Request

url='http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/Minerva/Minerva.QRP.settings/Settings_PARLOG/V9/parms/probes/lowerTestDivertorUnit/Probe_11/adc4/channels/channel7/modeResistor/_signal.json?from=1505088000000000000&upto=1505174400000000000'
json.loads(urlopen(url,timeout=pyfusion.TIMEOUT).read().decode('utf-8'))

{u'datatype': u'Double',
 u'dimensionCount': 1,
 u'dimensionSize': 2,
 u'dimensions': [1504788000000000000, 9223372036854775807],
 u'label': u'modeResistor',
 u'sampleCount': 2,
 u'unit': u'unknown',
 u'values': [10.0, 10.0]}


"""
