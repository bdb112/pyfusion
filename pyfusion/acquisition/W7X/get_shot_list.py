#!/usr/bin/env python
""" By extracting all progId, comments and dates, in ProjectDesc,
create a table to connect shots, dates and times.

Would be nice to have scenId and segId, but the relationship is not 1:1
To find segment 1 in scenario 2, shot xx need to find the utc range of shot xx,
find the scenarios in that utc, select 2, then find the segments within the
UTC range of the scenario, and identify segId 1.   Tricky.

This version reads both pickles and json versions of shotDA.  
Will probably make the transition to json only in the next version.

'rb' in load and dump works for both py2 and 3, but the files written by either can't be read by the other.
versions created in August are >1M due to the comments?
# 'rb' causes a problem with winpy - maybe using protocol 2 will fix?
# wrong ?? ->  under proto 2, need ,encoding='ascii' in python3 if file written by python2
# See test code at end
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys, json, time, os, calendar
import pyfusion


from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import codecs  # this is a workaround for urlopen/json - requests is supposed to be better

import pickle
if sys.version < '3, 0':     # 'fixups' for py2 compat.
    FileNotFoundError = IOError
    str = unicode

this_dir = os.path.dirname(__file__)

import json

#from pyfusion.acquisition.W7X.get_shot_info import get_shot_utc

def get_programs(shot=None, json_file='/data/datamining/local_data/W7X/json/all_programs.json'):
    """ (data.utc[0] - progs['programs'][-1]['trigger']['6'][0])/1e9 is the start of ECH
    Looks like '5' works better for 0309.052
    The original file was downloaded manually in three chunks.  
    http://archive-webapi.ipp-hgw.mpg.de/ has more details on downloading by machine

    After first call, caching reduces time to <20ms
    http://archive-webapi.ipp-hgw.mpg.de/programs.json
    http://archive-webapi.ipp-hgw.mpg.de/programs.json?from=1505174400000000000&upto=1505260799999999999
    """
    progs = json.load(open(json_file))
    programs = {}
    if shot is None or shot is -1:
        for p in progs['programs']:
            programs.update({p['id']: p})

    # check if it is the cached version - if not go to the web site
    if shot is not None: 
        decimal_shot = '{d}.{s:03d}'.format(d=shot[0], s=shot[1])
        if  decimal_shot not in programs:
            from pyfusion.acquisition.W7X.get_shot_info import get_shot_utc
            utc = get_shot_utc(shot) if shot[0] < 1e9 else shot # assume shot is utcs if very big
            prog_url = ("http://archive-webapi.ipp-hgw.mpg.de/programs.json?from={f}&upto={t}"
                        .format(f=utc[0], t=utc[1]))  #1505174400000000000&upto=1505260799999999999
            this_program = json.loads(urlopen(prog_url, timeout=pyfusion.TIMEOUT).read().decode('utf-8'))
            programs.update({decimal_shot: this_program['programs'][0]})

    return(programs)

def get_latest_program():
    """ meant to be a quck way to get only the latest programs, for use 
    during operation"""
    pass

def json_save_shot_list(shotDA, new_name='/tmp/shotDA.json'):
    shot_dict = {}
    for k in shotDA:
        if isinstance(shotDA[k], np.ndarray):
            shot_dict.update({k: shotDA[k].tolist()})
        else:
            shot_dict.update({k: shotDA[k]})

    json.dump(shot_dict, open(new_name, 'w'))


def _get_shotDA(fname):
    # TODO(boyd.blackwell@anu.edu.au): remove pickle option, use json only.
    ext = os.path.splitext(fname)[-1].lower()
    if pyfusion.VERBOSE>0: print('==> Trying ' +  fname)
    try:
        if ext == '.pickle':
            return(pickle.load(open(fname,'rb')))
    except (FileNotFoundError, UnicodeDecodeError, TypeError) as reason:
        if pyfusion.VERBOSE>0: print('Error opening {f}\n {r}'.format(f=fname, r=reason))
        return None

    if ext == '.json':
        # json.load() works here, but will follow example in 
        # http://webservices.ipp-hgw.mpg.de/docs/howtoREST.html#python, 
        # json file written by >>> json.dump(shotDA, file('shotDA.json', 'wt'))
        jd = json.loads(open(fname,'rt').read())
        shotDA = {}
        for k in jd:
            if isinstance(jd[k][0], str):
                shotDA.update({k: jd[k]})
            else:
                shotDA.update({k: np.array(jd[k])})
        return(shotDA)
    else:
        raise LookupError('file {f} cannot be read as a shotDA'.format(f=fname))

def get_shot_number(utcs):
    """ return the shot number covering a given time interval in ns utc (or None)
    """ 
    shotDA = get_shotDA()
    # try to find its shot if it has one.
    #    this_shot = [prog for prog in shotDA if (utcs[0] >= prog['from']
    #                                             and utcs[0] <= prog['upto'])]
    this_shot = [[shotDA['date'][i],shotDA['progId'][i]] for i in range(len(shotDA['date'])) if  (shotDA['start_utc'][i] <= utcs[0] and shotDA['end_utc'][i] >= utcs[0])]

    if len(this_shot) is 1:
        # return([this_shot[0]['date'], this_shot[0]['progId']])
        #return([shotDA['date'][this_shot[0]], shotDA['progId'][this_shot[0]]])
        return this_shot[0]
    else:
        return None
    
def get_shotDA(fname=os.path.join(this_dir, 'shotDA')):
    """ get the DA data file containing cached copy of the shot numbers, 
    utcs comments etc
    """
    if os.path.splitext(fname)[-1] != '':
        return _get_shotDA(fname)
    else:
        for ext in ['.json', '.pickle']:
            ret = _get_shotDA(fname + ext)
            if ret is not None:
                return(ret)
        raise LookupError('shotDA file {fname} not found'.format(fname=fname))

def test_pickles(from_file='/pyfusion/acquisition/W7X/shotDA.pickle', to_file='/tmp/x{v}.pickle'):
    """ aim to test compatibility of shotDA.pickle between python versions 
    not yet implemented - may discard if we use json files.
    """
    if '{' in to_file:
        to_file = to_file.format(v=sys.version[0:3])
    shotDA=pickle.load(open(os.path.join(this_dir,from_file),'rb'))
    pickle.dump(shotDA, open(to_file+'2','wb'),protocol=2)
    pickle.dump(shotDA, open(to_file+'1','wb'),protocol=1)

#### main ########
if __name__ == "__main__":
    _var_defaults="""
fr_UTC = '20150101 00:00:00'
to_UTC = '20180101 00:00:00'
seldate=0
seltext=""
update=0
"""

    exec(_var_defaults)
    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())

    # This %s format doesn't work under Windows, and is not in the docs
    #fr_utc = int(1e9 * int(time.strftime('%s', time.strptime(fr_UTC, '%Y%m%d %H:%M:%S'))))
    fr_utc = int(1e9) * int(calendar.timegm(time.strptime(fr_UTC, '%Y%m%d %H:%M:%S')))
    to_utc = int(1e9) * int(calendar.timegm(time.strptime(to_UTC, '%Y%m%d %H:%M:%S')))

    # for online access
    ArchiveDB = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/ProjectDesc.1/ProgramLabelLog/parms/{key}/_signal.json?from={fr_utc}&upto={to_utc}'
    #ArchiveDB = 'file:///data/databases/W7X/cache/{key}.json'  # for local files

    if update:
        reader = codecs.getreader("utf-8")

        shotinfo = {}
        # for each of the three variables, get a dictionary of values and times
        for itm in ['comment','ProgramLogLabel/progId','ProgramLogLabel/date']:
            key = itm.split('/')[-1]  # choose the last part of the name
            url = ArchiveDB.format(key=itm,fr_utc=fr_utc,to_utc=to_utc)
            shotinfo.update({key: json.load(reader(urlopen(url)))})

        # There should be a progId, comment and date for each shot.
        # and the date (or progId) should have two times - start and end - hence::2
        shotDA = {}
        for k in shotinfo:
            shotDA.update({k: shotinfo[k]['values'][::2]})

        shotDA.update({'start_utc': shotinfo['date']['dimensions'][::2]})
        shotDA.update({'end_utc': shotinfo['date']['dimensions'][1::2]})

        # change the integer types to numpy int64 arrays for safety
        for k in ['date','progId', 'start_utc', 'end_utc']:
          shotDA.update({k: np.array(shotDA[k], np.int64)})  # make it an array of ints

        # gather a few statistics to print, and a day's comments
        d = np.unique(shotDA['date'])
        wL = np.where((shotDA['end_utc']-shotDA['start_utc'])>60e9)[0]
        print('Over {d} days, there were {n} shots, {N} longer than 100ms'
              .format(N=len(wL), d=len(d), n=len(shotDA['date'])))
        pickle.dump(shotDA, open(os.path.join(this_dir,'shotDA.pickle'),'wb'),protocol=2)
        json_save_shot_list(shotDA, new_name=os.path.join(this_dir,'shotDA.json'))
        
    else:
        shotDA = get_shotDA()

    if seldate is 0 and seltext is "":
        seldate = max(shotDA['date'])
    if seltext != "":
        # this doesn't work
        # selected = np.where(seltext in shotDA['comment'])[0]
        selected = []
        for (i,com) in enumerate(shotDA['comment']):
            if seltext.lower() in com.lower():
                selected.append(i)

    else:
        selected = np.where(seldate == shotDA['date'])[0]

    for i in selected:
        print(u'<<< {dtm}, {progId}, {tm} {dt}\n{cmt} >>>' 
              .format(dtm=shotDA['date'][i],progId=shotDA['progId'][i],
                      tm=time.asctime(time.gmtime(shotDA['end_utc'][i]/1e9)),
                      dt=1e-9*(shotDA['end_utc'][i]-shotDA['start_utc'][i]),
                      cmt=shotDA['comment'][i].replace('\n',' ')))

