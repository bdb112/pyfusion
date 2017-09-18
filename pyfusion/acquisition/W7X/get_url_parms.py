""" Obtain a dictionary containing all the TDU and Striker Plate Langmuir probe data

"""
from __future__ import print_function
from future.standard_library import install_aliases
from collections import OrderedDict
import json
import sys
import numpy as np

import pyfusion
from pyfusion.acquisition.H1.scrape_wiki import get_links
from pyfusion.debug_ import debug_

install_aliases()
from urllib.request import urlopen, Request

class MinervaMap(object):
    def __init__(self):
        fname =  "http-__archive-webapi.ipp-hgw.mpg.de_ArchiveDB_raw_Minerva_Minerva.QRP.settings_Settings_PARLOG_V12.json"
        #'http-__archive-webapi.ipp-hgw.mpg.de_ArchiveDB_raw_Minerva_Minerva.QRP.settings_Settings_PARLOG_V11.json'
        self.parmdict = json.load(file(fname, 'rt'))

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
    mm = MinervaMap()
    print('last mod at ', mm.get_parm('parms/modifiedAt/values')[0], mm.get_parm('parms/generalRemarks/values')[0] )
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
        rs = None
        gainstr = str((electronics['driverElectronicTotalGain']['values'][0])) + ' V'

    fetcher_obj.gain = gainstr
    dmd = 180 + adc_no  # adc1 is the left most, and maps to 181_DATASTREAM
    if adc_no > 4: dmd += 4
    
    fetcher_obj.params = "CDS=82,DMD={dmd},ch={ch}".format(dmd=dmd, ch=chan_no)
    debug_(pyfusion.DEBUG, 1, key='MinervaName', msg='leaving MinervaName')
    print('rs={rs}, overall gain={gain}, params={params}'
          .format(rs=rs, gain=fetcher_obj.gain, params=fetcher_obj.params))
    return(fetcher_obj)


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


if __name__ == '__main__':
    URL = sys.argv[1] if len(sys.argv) > 1 else 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/Minerva/Minerva.QRP.settings/Settings_PARLOG/V12'
    parmdict = get_subtree(URL)
    fname = URL.replace('/','_').replace(':','-') + '.json'
    print('saving in ', fname)
    json.dump(parmdict, file(fname, 'wt'))


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
