#!/usr/bin/env python
""" By extracting all progId, comments and dates, in ProjectDesc,
create a table to connect shots, dates and times.

Then with keywords seldate, selshot, seltext, print the selected shot information

e.g.         # to show all shots 21 (handy to see dates with data taken)
  run pyfusion/acquisition/W7X/get_shot_list.py selshot=21
  run pyfusion/acquisition/W7X/get_shot_list.py seldate=20180911
  run pyfusion/acquisition/W7X/get_shot_list.py seltext='std'

Note that comments (case insensitive) are not used much anymore

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

Example to check relative timing of triggers.  Can't make sense of it for 2016 shots.
A comment elsewhere suggests to use '6'
[[pr, progs[pr]['trigger']['5'][0] - progs[pr]['trigger']['1'][0]] for pr in progs if progs[pr]['trigger'] is not None and len(progs[pr]['trigger']['1'])>0]


"""

from __future__ import print_function
import builtins

import numpy as np
import matplotlib.pyplot as plt
import sys, json, time, os, calendar
import pyfusion
from pyfusion.utils.time_utils import utc_ns, utc_GMT
from pyfusion.debug_ import debug_

from future.standard_library import install_aliases
install_aliases()

from pprint import pprint
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
from copy import deepcopy
def make_utcs_relative_seconds(dic, t0):
    """ Take a dictionary and replace all utcs with seconds relative to t0
    This helps with debugging in general and understanding programs.
    """
    dic = deepcopy(dic)
    lst = (tuple, list, np.ndarray)
    anyints = (builtins.int, np.int32, np.int64)
    for k, val in dic.iteritems():
        if isinstance(val, anyints) and val > 1e9:
            dic.update({k: (val - t0)/1e9})
        elif isinstance(val, lst):
            vnew = [(v - t0)/1e9 if isinstance(v, anyints) and v > 1e9
                    else v for v in val[0:10]]  # prevent long lists from floding screen
            dic.update({k: vnew})
    return dic

def pprint_params(dic, utc_or_shot=None):
    """ pretty print a dictionary, showing ns times relative to t0, which can be given as 
           a utc, shot in utc or date, number (W7X)
           e.g. pprint_params(signal_dict['params'],[20160310,11])
                pprint_params(data.params,[20160310,11])   
    Note:  this version had only been run on ipp VM
    To overcome absence of program data (making all relative to utc[0])
        pprint_params(data.params,data.params['utc'][0])

    >>> offset = 1000000000*123456; parms = dict(utc0=offset, utc1=offset+10**9, deci=1.23, xstr='unchanged')
    >>> pprint_params(parms, 0); pprint_params(parms, offset)
    {'deci': 1.23, 'utc0': 123456.0, 'utc1': 123457.0, 'xstr': 'unchanged'}
    {'deci': 1.23, 'utc0': 0.0, 'utc1': 1.0, 'xstr': 'unchanged'}
    """
    if isinstance(utc_or_shot, list):
        if utc_or_shot[1] < 1e9:
            progs = get_programs(utc_or_shot)
            t0 = progs.values()[0]['from']
        else:
            t0 = utc_or_shot[0]
    elif utc_or_shot is None:
        if 'utc_0' in dic:
            print('>>> relative to utc_0')
            utc_or_shot = dic['utc_0']
        elif 'shot_f' in dic:
            print('>>> relative to shot start')
            utc_or_shot = dic['shot_f']
        else:
            print('No suitable time reference in ', dic.keys(), '\n try setting utc_or_shot')
            raise ValueError('No suitable time reference in given dictionary')
    t0 = utc_or_shot
    
    copydic = deepcopy(dic)
    reldic = make_utcs_relative_seconds(copydic, t0)
    pprint(reldic)
    
def show_program(shot=[20181018,41]):  # pretty print program times
    progs = get_programs(shot)
    assert len(progs) == 1, 'no or too many programs matching'
    prog = deepcopy(progs.values()[0])
    trel = prog['from']
    conv_prog = make_utcs_relative_seconds(prog, trel)
    prog = conv_prog
    triggers = prog['trigger']
    prog.update(dict(trigger=make_utcs_relative_seconds(triggers, trel)))
    scenarios = prog['scenarios']
    assert isinstance(scenarios, list), 'scenarios is not a list'
    newscens = [make_utcs_relative_seconds(scen, trel) for scen in scenarios]
    prog.update(dict(scenarios=newscens))
    pprint(prog)
    
def get_programs(shot=None, json_file='/data/datamining/local_data/W7X/json/all_programs.json'):
    """ 
    get the program for a shot, or reload the caches (shot = None or -1)
    (data.utc[0] - progs['programs'][-1]['trigger']['6'][0])/1e9 is the start of ECH
    Looks like '5' works better for 0309.052
    The original file was downloaded manually in three chunks.  
    http://archive-webapi.ipp-hgw.mpg.de/ has more details on downloading by machine

    After first call, caching reduces time to <20ms
    http://archive-webapi.ipp-hgw.mpg.de/programs.json
    http://archive-webapi.ipp-hgw.mpg.de/programs.json?from=1505174400000000000&upto=1505260799999999999
    """
    programs = {}

    if (shot is None or shot is -1) and os.path.exists(json_file):
        progs = json.load(open(json_file))
        for p in progs['programs']:
            programs.update({p['id']: p})

    # check if it is in the cached version - if not go to the web site
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
    """ meant to be a quick way to get only the latest programs, for use 
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
    global shotDA
    #shotDA = pyfusion.shotDA - Used a global instead. However that is probably be better..BUT need to copy dict!!!!
    try:
        #if shotDA is None:
        #    raise ValueError('shotDA is None - not defined')
        return(shotDA)
    except NameError:
        print('will load shotDA cache from ', fname)
    except Exception as reason:  
        print('Error accessing shotDA cache', reason)
        pass
    ext = os.path.splitext(fname)[-1].lower()
    if pyfusion.VERBOSE>0: print('==> Trying ' +  fname)
    try:
        if ext == '.pickle':
            shotDA = pickle.load(open(fname,'rb'))
            return(shotDA)
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


def get_standalone(event=None, utc_from=None, utc_upto=None, fname=os.path.join(this_dir, 'all_standalone.txt'), as_shot_list=True, dt=1, start=6):
    """ given a text file all_standalone.txt, return either
           an OrderedDict containing comment, event, leader for each key (ascii time)
           or if as_shot_list ==  True
           a time ordered shot list in utc_ns form, with delay of start, and length of dt
           Ken is asking if there is a programmatic way...
    """
    if utc_from is None:
        utc_from = utc_ns('20100101')
    if utc_upto is None:
        utc_upto = utc_ns('21000101')

    with open(fname) as fp:
        stdalbuf = fp.readlines()
        from collections import OrderedDict
        stdal = OrderedDict()
    for buf in stdalbuf[::-1]:
        buf = buf.strip()
        if buf == '':
            continue
        toks = buf.split('QRP', 1)  # maxsplit=1
        asc_time = toks[0].strip()
        if utc_ns(asc_time) < utc_from or utc_ns(asc_time) > utc_upto:
            continue
        subtoks = toks[1].split(' ', 1)
        this_event = 'QRP' + subtoks[0]
        comment = subtoks[1].split('Session Leader')
        if len(comment) == 2:
            comment, leader = comment
        else:
            leader = ''
        stdal.update({this_event: dict(asc_time=asc_time, comment=comment, leader=leader)})

    debug_(pyfusion.DEBUG, 1, key='get_standalone')
    if event is not None:
        stdal = {event: stdal[event]}
    if as_shot_list:
        stdal = np.sort([utc_ns(dic['asc_time']) for dic in stdal.itervalues()])
        stdal = [[ns + int(start * 1e9), ns + int((start + dt) * 1e9)] for ns in stdal]
    return(stdal)


def test_pickles(from_file='/pyfusion/acquisition/W7X/shotDA.pickle', to_file='/tmp/x{v}.pickle'):
    """ aim to test compatibility of shotDA.pickle between python versions 
    not yet implemented - may discard if we use json files.
    """
    if '{' in to_file:
        to_file = to_file.format(v=sys.version[0:3])
    shotDA = pickle.load(open(os.path.join(this_dir, from_file), 'rb'))
    pickle.dump(shotDA, open(to_file+'2', 'wb'), protocol=2)
    pickle.dump(shotDA, open(to_file+'1', 'wb'), protocol=1)


def reduce(f, t, delta=4000):
    """ combine adjacent segments
    reduce([11,9,7,1],  [12,10,8,3], delta=1)
`        (array([7, 1]), array([12,  3]))
    """
    froms = [f.pop(0)]
    tos = [t.pop(0)]
    while len(f) > 0:
        while froms[-1] == t[0] + delta:
            froms[-1] = f.pop(0)
            t.pop(0)
            if len(f) == 0:
                break
        if len(f) == 0:
            break
        froms.append(f.pop(0))
        tos.append(t.pop(0))
    return(np.array(froms, dtype=np.int64), np.array(tos, dtype=np.int64))


def get_segments(seg_list_file=None, reduced=1):
    """
    plot(get_segments()['froms'],(get_segments()['upto']-get_segments()['froms'])/1e9,',') ; ylim(-1,10)
    """
    
    if seg_list_file is None:
        seg_list_file = os.path.join(this_dir, 'seg_list.json')
        seg_dict = json.load(open(seg_list_file, 'r'))
        seg_list = seg_dict['seg_list']
        
    froms = [int(seg['href'].split('from=')[1].split('&upto=')[0]) for seg in seg_list]
    upto  = [int(seg['href'].split('upto=')[1].strip()) for seg in seg_list]
    if reduced:
        froms, upto = reduce(froms, upto)
    return(dict(froms=froms, upto=upto))


# ##### main ###
if __name__ == "__main__":
    _var_defaults = """
fr_UTC = '20150101 00:00:00'
to_UTC = '20190101 00:00:00'
seldate=0
seltext=""
selshot=0
update=0
"""

    exec(_var_defaults)
    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())

    # This %s format doesn't work under Windows, and is not in the docs
    # fr_utc = int(1e9 * int(time.strftime('%s', time.strptime(fr_UTC, '%Y%m%d %H:%M:%S'))))
    fr_utc = int(1e9) * int(calendar.timegm(time.strptime(fr_UTC, '%Y%m%d %H:%M:%S')))
    to_utc = int(1e9) * int(calendar.timegm(time.strptime(to_UTC, '%Y%m%d %H:%M:%S')))
    now =  int(1e9) * calendar.timegm(calendar.datetime.datetime.utcnow().utctimetuple())
    if now > to_utc:
        pyfusion.utils.warn('**** Warning - you are not capturing current data - only up to '+ to_UTC)

    # for online access
    ArchiveDB = 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/ProjectDesc.1/ProgramLabelLog/parms/{key}/_signal.json?from={fr_utc}&upto={to_utc}'
    # ArchiveDB = 'file:///data/databases/W7X/cache/{key}.json'  # for local files

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

    if seldate == 0 and seltext == "" and selshot == 0:
        seldate = max(shotDA['date'])
    if seltext != "":
        # this doesn't work
        # selected = np.where(seltext in shotDA['comment'])[0]
        selected = []
        for (i,com) in enumerate(shotDA['comment']):
            if seltext.lower() in com.lower():
                selected.append(i)

    elif seldate > 0:
        selected = np.where(seldate == shotDA['date'])[0]
    elif selshot > 0:
        selected = np.where(selshot == shotDA['progId'])[0]
    else:
        raise ValueError('Must have selshot, seldate, or seltext non 0/""')

    for i in selected:
        print(u'<<< {dtm}, {progId}, {tm} {dt}{nl}{cmt} >>>' 
              .format(dtm=shotDA['date'][i],progId=shotDA['progId'][i],
                      tm=time.asctime(time.gmtime(shotDA['end_utc'][i]/1e9)),
                      dt=1e-9*(shotDA['end_utc'][i]-shotDA['start_utc'][i]),
                      cmt=shotDA['comment'][i].replace('\n',' '),
                      nl = '\n' if len(shotDA['comment'][i]) > 50 else ' '))

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # not sure why  running this prints 41 shots
