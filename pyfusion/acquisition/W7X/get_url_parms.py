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

install_aliases()
from urllib.request import urlopen, Request

class MinervaMap(object):
    def __init__(self, shot):
        # fname =  "http-__archive-webapi.ipp-hgw.mpg.de_ArchiveDB_raw_Minerva_Minerva.QRP.settings_Settings_PARLOG_V25_.json"
        parm_URL = get_suitable_version(shot)
        fname = sanitised_json_filename(parm_URL)

        try:
            self.parmdict = json.load(open(fname, 'rt'))
        except FileNotFoundError:
            print('run pyfusion/acquisition/W7X/get_url_parms.py  # to generate minerva cache')

    def get_parm(self, pseudourl=None):
        return get_parm(pseudourl, self.parmdict)
                
    
def get_parm(pseudourl, parmdict):
    """ Example: to get probe resistance
    get_parm('parms/probes/upperTestDivertorUnit/Probe_2/adc1/channels/channel2/modeResistor/values', parmdict)[0]

    """
    if pseudourl.endswith('/'):
        pseudourl = pseudourl[0:-1]

    subdict = parmdict.copy()
    for key in pseudourl.split('/'):
        subdict = subdict[key]

    return(subdict)



def get_minerva_parms(fetcher_obj):
    mm = MinervaMap(fetcher_obj.shot)
    last_mod = mm.get_parm('parms/modifiedAt/values')[0]
    cal_remarks = mm.get_parm('parms/generalRemarks/values')[0] 
    print('last mod at ', last_mod, cal_remarks)
    adcs = list(mm.get_parm('parms/probes/{mn}'.format(mn=fetcher_obj.minerva_name)))
    if len(adcs) == 0:
        raise LookupError('No adcs found for {minerva_name} on {shot}'
                          .format(mn=fetcher_obj.minerva_name, shot=fetcher_obj.shot))
    elif len(adcs) == 1:
        if fetcher_obj.config_name.endswith('I'):
            adc = adcs[0]
        else:
            raise LookupError('No monitor on ' + fetcher_obj.config_name)

    elif len(adcs) == 2:
        adcs = np.sort(adcs)
        if fetcher_obj.config_name.endswith('I'):
            adc = adcs[0]
        else:
            adc = adcs[-1]
    else:
        debug_(pyfusion.DEBUG, 1, key='MinervaName', msg='leaving MinervaName')
        raise ValueError('Too many adcs found for {mn} on {shot}'
                          .format(mn=fetcher_obj.minerva_name, shot=fetcher_obj.shot))
            
    chans = list(mm.get_parm('parms/probes/{mn}/{adc}/channels'.format(mn=fetcher_obj.minerva_name, adc=adc)))
    if len(chans) != 1:
        raise LookupError('chan not found for {mn} on {shot}, adc {adc}'
                          .format(mn=fetcher_obj.minerva_name, shot=fetcher_obj.shot, adc=adc))
    chan = chans[0]
    electronics = mm.get_parm('parms/probes/{mn}/{adc}/channels/{chan}'
                              .format(mn=fetcher_obj.minerva_name, adc=adc, chan=chan))
    chan_no = int(chan[7:]) - 1
    adc_no = int(adc[3:])
    # gain and params are in string representation in the OP1.1, so do the same here
    if fetcher_obj.config_name.endswith('I'):
        rs = electronics['modeResistor']['values'][0]
        gainstr = str(1/(rs*electronics['driverElectronicTotalGain']['values'][0])) + ' A'
    else:
        modeFactor = electronics['modeFactor']['values'][0]
        rs = None
        gainstr = str(1/(modeFactor*electronics['driverElectronicTotalGain']['values'][0])) + ' V'

    fetcher_obj.gain = gainstr
    dmd = 180 + adc_no  # adc1 is the left most, and maps to 181_DATASTREAM
    if adc_no > 4: dmd += 4
    
    fetcher_obj.params = "CDS=82,DMD={dmd},ch={ch}".format(dmd=dmd, ch=chan_no)
    debug_(pyfusion.DEBUG, 1, key='MinervaName', msg='leaving MinervaName')
    print('rs={rs}, overall gain={gain}, params={params}'
          .format(rs=rs, gain=fetcher_obj.gain, params=fetcher_obj.params))
    return(fetcher_obj, dict(cal_date=last_mod, cal_comment=cal_remarks))


def get_subtree(url, debug=1, level=0):
    """ level is just for pretty printing
    """
    dic = OrderedDict()   # Don't forget the parens, Boyd
    links = get_links(url, debug=0)
    for lnk, name in links:
        name = name.strip()
        if url not in lnk or 'Remove' in name:
            continue
        if lnk.endswith('/'):
            if debug > 0: print('recursing into ', level*'    ', lnk.replace(url,'-'), name)
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

def get_signal_url(path='CBG_ECRH/A1/medium_resolution/Rf_A1'):
    """ return the cryptic url corresponding to the human readable form """
    import os
    root = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/views/KKS/'
    oneup, wantname = os.path.split(path)
    fullurl = os.path.join(root, oneup)
    links = get_links(fullurl, debug=0)
    for lnk, name in links:
        name = name.strip()
        if 'DATASTREAM' not in lnk or 'Remove' in name:
            continue
        if name == wantname:
            print('Found!')
            return(lnk)
    raise LookupError(path + ' not found in ' + root)


def get_PARMLOG_cache(shot):
    """ this is very similar to get_suitable params, but uses a local file cache
    The cache should be persistent across 'run' commands - an easy way
    is to initialise in pyfusion.
 """
    Vfiles = glob.glob("*PARLOG_V[0-9]*json")
    # put them in a dictionary as a cache - should live in an object
    # (maybe MinervaMap, but save in pyfusion,,,, for now)
    if not hasattr(pyfusion, 'W7X_minerva_cache') or pyfusion.W7X_minerva_cache is None:
        print('loading pyfusion.W7X_minerva_cache ..')
        pyfusion.W7X_minerva_cache = dict([[fn, json.load(open(fn, 'rt'))] for fn in Vfiles])

    vnumstrs = [vfile.split('PARLOG_V')[-1].split('_.json')[0] for vfile in Vfiles]
    vnums  = np.array([int(vn) for vn in vnumstrs if vn is not None])
    sortd = np.argsort(vnums)[::-1]
    utcs = get_shot_utc(shot)
    # now work down thourgh the versions and find the first where shot utc is after valid
    for vkey in np.array(Vfiles)[sortd]:
        debug_(pyfusion.DEBUG, 1, key='get_PARMLOG_cache', msg='reading PARLOG_V files')
        if get_parm('parms/validSince/dimensions',pyfusion.W7X_minerva_cache[vkey])[0] < utcs[0]:
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
