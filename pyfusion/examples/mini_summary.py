"""  to be adapted to pyfusion from  a version hard-coded for H-1

General pyfusion version of mini_summary - see mini_summary_MDS for MDSPlus specific version

cd c:/cygwin/home/bobl/pyfusion/working/pyfusion
echo 'y' | python pyfusion/examples/mini_summary.py shot_list.txt

This version is even more robust - an error looking up
one diagnostic won't prevent data from being gathered for the others.
5/sec h1svr 2016
8/sec 19 diags to SSD file (/tmp) t440p 2016
12/sec 19 diags to memory or to file using no Journaling
10/sec using WAL only, 12.5? sec using WAL and large cache and synch off

In this example, the table is defined first

Example:  result = conn.execute('select shot, im3 from summ where im3>2000')
result.fetchone()
#_PYFUSION_TEST_@@Skip
# Jul-Dec 2015 progress report
result=conn.execute('select count(*) from summ where shot> 83808 and (isweep>2 or i_sat_1>0.2 or i_sat_2>0.2 or (mirnov_coh>0.02 and im2>2000))')

This works on W7-X and H-1
result=conn.execute('select * from summ limit 2')
result.fetchall()

Advanced: plotting
sh,k_h = np.transpose(conn.execute('select shot, is2/im2 as kh from summ where im2>2000').fetchall())
# one liner
plot(*(np.transpose(conn.execute('select shot, is2/im2 as kh from summ where im2>2000').fetchall()).tolist()),marker=',',linestyle=' ') 
scatter(*(np.transpose(conn.execute('select shot, is2/im2 as kh from summ where im2>2000').fetchall()).tolist()),marker='.')  # , not known to scatter


duds 36362
"""
from __future__ import print_function
import os
import sys
from pyfusion.data.signal_processing import smooth_n
import time as tm
from time import time as seconds
import pyfusion
from six.moves import input
import json
from pyfusion.utils import process_cmd_line_args, pause_while
from pyfusion.data.DA_datamining import report_mem
from matplotlib.mlab import specgram   # this is a non-plotting version
 
from sqlalchemy import create_engine 

devname = 'W7X'
if 'W7' in devname:
    sqlfilename = 'W7X_OP1.2ab_MHD_4124321436_rev.sqlite'
else:
    sqlfilename = 'H1.sqlite'

# put the file in the example (this) folder although it may be better in the acquisition/W7X folder
dbpath = os.path.dirname(__file__)
dbfile = os.path.join(dbpath, sqlfilename)
#engine=create_engine('sqlite:///:memory:', echo=False)
engine=create_engine('sqlite:///'+ dbfile, echo=False)
#engine = create_engine('mysql://127.0.0.1/bdb112', echo=False)
"""
from sqlalchemy.interfaces import PoolListener

from sqlalchemy import create_engine 

class MyListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        #dbapi_con.execute('pragma journal_mode=WAL')  # newer form  
        dbapi_con.execute('pragma journal_mode=OFF')
        dbapi_con.execute('PRAGMA synchronous=OFF')
        dbapi_con.execute('PRAGMA cache_size=100000')

basefile = '/tmp/sqltest'
basefile = ':memory:'

engine = create_engine('sqlite:///' + basefile,echo=False, listeners= [MyListener()])
"""
conn = engine.connect()
import numpy as np

from sqlalchemy import Table, Column, Integer, String, Float, MetaData#, ForeignKey
from pyfusion.data.shot_range import shot_range as expand_shot_range
from collections import OrderedDict as ordict

# simple method - use a fixed interval : typical H-1 [0.01, 0.05]
# here 
initial_interval = [0.01, 0.05] # [0,10] 
interval = initial_interval

############
## Utilities
def mini_dump(list, filename='mini_summary_{0}_{1}.pickle'):
    """ dump in the form of a pickle, filename optionally includes beginning and ending shots
    """
    import cPickle as pickle
    try:
        list.append(conn.execute('select * from sqlite_master').fetchall())
    except:
        print('skipped appending the schema')
    fnamefull = filename.format(*conn.execute('select min(shot), max(shot) from summ').fetchone())
    print('dumping to ' + fnamefull)
    pickle.dump(list,file(fnamefull,'w'))

###########################
# Functions to process data
           
def Peak_abs(x, t, pc=100, return_time=False):
    frac = (100 - pc) / 100.
    disc = int(round(frac * len(x)))
    xa = np.abs(x)
    max_idx = np.argsort(xa)[-1 - disc]
    if return_time:
        if t is None:
            return None
        else:
            return(t[max_idx])
    else:
        return xa[max_idx]
    
def rawPeakSPD(x, t=None, NFFT=1024):  # A power, not amplitude
    return(np.max(specgram(x, NFFT=NFFT, noverlap=0)[0].flatten()))

def rawPeakSPDtime(x, t=None, NFFT=1024):
    if t is None:
        return None
    spd = specgram(x, NFFT=NFFT, noverlap=0, Fs=1/np.diff(t).mean() )
    # Use flatten tomake sure that both dimensions are scanned
    idxs = np.unravel_index(np.argmax(spd[0].flatten()), spd[0].shape)
    return(spd[2][idxs[1]] + t[0])  # this time will not exactly be in the timebase

def Peak_signed(x, t, pc=100, return_time=False):
    """ returns the largest in magnitude but with the original sign """
    frac = (100 - pc) / 100.
    disc = int(round(frac * len(x)))
    # unnecessary complication - just operate on abs
    extr_idx = [np.argsort(x)[off] for off in [disc, -1 - disc]]
    greater_idx = np.argsort(np.abs([x[ind] for ind in extr_idx]))[-1]
    max_idx = extr_idx[greater_idx]
    if return_time:
        if t is None:
            return None
        else:
            return(t[max_idx])
    else:
        return x[max_idx]
    
def rawPeak(x, t=None):
    """ the highest """
    return np.sort(x)[-1]

def rawPeaktime(x, t=None):
    """ the highest """
    return None if t is None else t[np.argsort(x)[-1]]

def rawPeak(x, t):
    """ 100th percentile """
    return(Peak_abs(x, t, pc=100))

def rawPeaktime(x, t):
    """ 100th percentile """
    return(Peak_abs(x, t, pc=100, return_time=True))           

def Peak(x, t):
    """ 99th percentile """
    return(Peak_abs(x, t, pc=99))

def Peaktime(x, t):
    """ 99th percentile """
    return(Peak_abs(x, t, pc=99, return_time=True))           

def fPeak(x, t):
    """ 97th percentile """
    return(Peak_abs(x, t, pc=97))

def fPeaktime(x, t):
    """ 97th percentile """
    return(Peak_abs(x, t, pc=97, return_time=True))           

def Special():
    # hard coded for now
    pass


def Std(x, t=None):
    return(np.std(x))

def Average(x, t=None):
# should really to an average over an interval
    if t is None:
        t = x.timebase
        x = x.signal
    w = np.where((t>interval[0]) & (t>interval[1]))[0]
    return np.average(x[w])

def Integral(x, t):
# should really include dt
    return np.average(np.diff(t))*np.sum(x)

def Value(x, t):
    if np.issubdtype(type(x), int):
        return(int(x))
    else:
        return(x)

metadata = MetaData()

# main
debug = 0  #  For looping, debug=1 (stops on diag node exception)
#             might be better to avoid shots being labelled as bad
took = 0
#pyfusion.NSAMPLES=2000


if len(sys.argv)>1:
    print('Assuming srange is given in the command line, and ranges start with [ or (')
    try:
        srangestr = sys.argv[1] 
        if srangestr.startswith('[') or srangestr.startswith('('):
            srange = eval(srangestr)
        else:
            shot_file = os.path.join(dbpath, srangestr)
            # json.dump(shot_list, open(os.path.join(dbpath, 'shot_list.txt'), 'w'))
            srange = json.load(open(shot_file, 'r'))
    except Exception as reason:
        raise ValueError("Exception: {r}: \nInput was\n{inp}\nProbably need quotes:\n  Example:\n  run pyfusion/examples/mini_summary 'range(88600,88732)' or  [[20171018,19],[20171018,21]] - no quotes needed if square brackets"
                             .format(r=reason, inp=sys.argv))
else:
    srange = range(88600,88732)
    srange = ((20160309,1), (20160310,99))
    srange = ((20160101,1), (20160310,99))
    srange = ((20160101,1), (20171110,99))
    #srange = ((20160202,1), (20160202,99))
    srange = ((20171018,19),(20171018,20))
    srange = ((20180801,1),(20180822,999))
    srange = ((20171207,1),(20181022,999))
    srange = (([20171207,1], [20171231,999]),([20180701,1], [20181022,999]))

    
MDS_diags = dict(im1 = [Float, '.operations.magnetsupply.lcu.setup_main.i1', Value],
             im2 = [Float, '.operations.magnetsupply.lcu.setup_main.i2', Value],
             im3 = [Float, '.operations.magnetsupply.lcu.setup_main.i3', Value],
             is1 = [Float, '.operations.magnetsupply.lcu.setup_sec.i1', Value],
             is2 = [Float, '.operations.magnetsupply.lcu.setup_sec.i2', Value],
             is3 = [Float, '.operations.magnetsupply.lcu.setup_sec.i3', Value],
             rf_drive = [Float, '.rf.rf_drive', Peak],
             i_fault = [Float, '.operations:i_fault', Average],
             rf_freq = [Float, '.log.heating.rf_freq', Value],
             llrf_mode = [Integer, '.log.heating.snmp.t1.operational.llrf.stallrfopm', Value],
             llrf_power = [Float, '.log.heating.snmp.t1.operational.llrf.stallrfppw', Value],
             llrf2_mode = [Integer, '.log.heating.snmp.t2.operational.llrf.stallrfopm', Value],
             llrf2_power = [Float, '.log.heating.snmp.t2.operational.llrf.stallrfppw', Value],
             v_main =  [Float, '.operations:v_main', Peak],
             int_v_main =  [Float, '.operations:v_main', Integral],
             v_sec =  [Float, '.operations:v_sec', Peak],
             int_v_sec =  [Float, '.operations:v_sec', Integral],
             i_sat_1 = [Float, '.fluctuations.BPP:vf_pin1',Std],
             i_sat_2 = [Float, '.fluctuations.BPP:vf_pin2',Std],
             isweep = [Float, '.fluctuations.BPP:isweep',Std],
             mirnov_coh =  [Float, '', Special],
         )
# Magnetic and basic plasma params
diags = dict(imain = [Float, 'W7X_INonPlanar_1', Average, 1], # the first will be used in the example code
             IPlanar_A = [Float, 'W7X_IPlanar_A', Average, 1],
             IPlanar_B = [Float, 'W7X_IPlanar_B', Average, 1],
             INonPlanar_1 = [Float, 'W7X_INonPlanar_1', Average, 1],
             INonPlanar_2 = [Float, 'W7X_INonPlanar_2', Average, 1],
             INonPlanar_3 = [Float, 'W7X_INonPlanar_3', Average, 1],
             INonPlanar_4 = [Float, 'W7X_INonPlanar_4', Average, 1],
             INonPlanar_5 = [Float, 'W7X_INonPlanar_5', Average, 1],
             ITrim_1 = [Float, 'W7X_ITrim_1', Average, 1],
             ITrim_2 = [Float, 'W7X_ITrim_2', Average, 1],
             ITrim_3 = [Float, 'W7X_ITrim_3', Average, 1],
             ITrim_4 = [Float, 'W7X_ITrim_4', Average, 1],
             ITrim_5 = [Float, 'W7X_ITrim_5', Average, 1],
             ICtl_1U = [Float, 'W7X_ICtl_1U', Std, 1],
             ICtl_1L = [Float, 'W7X_ICtl_1L', Std, 1],
             ICtl_2U = [Float, 'W7X_ICtl_2U', Std, 1],
             ICtl_2L = [Float, 'W7X_ICtl_2L', Std, 1],
             ICtl_3U = [Float, 'W7X_ICtl_3U', Std, 1],
             ICtl_3L = [Float, 'W7X_ICtl_3L', Std, 1],
             ICtl_4U = [Float, 'W7X_ICtl_4U', Std, 1],
             ICtl_4L = [Float, 'W7X_ICtl_4L', Std, 1],
             ICtl_5U = [Float, 'W7X_ICtl_5U', Std, 1],
             ICtl_5L = [Float, 'W7X_ICtl_5L', Std, 1],
             TotECH = [Float, 'W7X_TotECH', Peak, 1],
             WDIA_TRI = [Float, 'W7X_WDIA_TRI', Peak, 1],
             IP_CONT = [Float, 'W7X_ROG_CONT', Peak, 1],
             V_SWEEP_RMS = [Float, 'W7X_KEPCO_U', Std, 1],
             V_SWEEP_AVG = [Float, 'W7X_KEPCO_U', Average, 1],
             # partial, and needs more care peak or average (spikes?)
             #ECH_Rf_A1 = [Float, 'W7X_ECH_Rf_A1', Average, 1],
             #ECH_Rf_B1 = [Float, 'W7X_ECH_Rf_B1', Average, 1],
             #ECH_Rf_C1 = [Float, 'W7X_ECH_Rf_C1', Average, 1],
             #ECH_Rf_D1 = [Float, 'W7X_ECH_Rf_D1', Average, 1],
             #ECH_Rf_A5 = [Float, 'W7X_ECH_Rf_A5', Average, 1],
             #ECH_Rf_B5 = [Float, 'W7X_ECH_Rf_B5', Average, 1],
             #ECH_Rf_C5 = [Float, 'W7X_ECH_Rf_C5', Average, 1],
             #ECH_Rf_D5 = [Float, 'W7X_ECH_Rf_D5', Average, 1],
             )
# Scraper study
diags = ordict(iA = [Float, 'W7X_IPlanar_A', Average, 1],
               iB = [Float, 'W7X_IPlanar_B', Average, 1],
               i1 = [Float, 'W7X_INonPlanar_1', Average, 1],
               i2 = [Float, 'W7X_INonPlanar_2', Average, 1],
               i3 = [Float, 'W7X_INonPlanar_3', Average, 1],
               NBI4_I = [Float, 'W7X_NBI4_I', fPeak, 1],
               NBI4_U = [Float, 'W7X_NBI4_U', fPeak, 1],
               scr_1 = [Float, 'W7X_STDU_LP01_I', Std, 3],
               scr_2 = [Float, 'W7X_STDU_LP02_I', Std, 3],
               scr_3 = [Float, 'W7X_STDU_LP03_I', Std, 3],
               scr_4 = [Float, 'W7X_STDU_LP04_I', Std, 3],
               scr_5 = [Float, 'W7X_STDU_LP05_I', Std, 3],
               scr_6 = [Float, 'W7X_STDU_LP06_I', Std, 3],
               scr_7 = [Float, 'W7X_STDU_LP07_I', Std, 3],
               scr_8 = [Float, 'W7X_STDU_LP08_I', Std, 3],
               TotECH = [Float, 'W7X_TotECH', Peak, 1],
               WDIA_TRI = [Float, 'W7X_WDIA_TRI', Peak, 1],
               IP_CONT = [Float, 'W7X_ROG_CONT', Peak, 1],
               neL = [Float, 'W7X_neL', Peak, 1],
)
# for MHD study
diags = ordict()

diags.update(dict(PkrSPD4124time = [Float, 'W7X_MIR_4124', rawPeakSPDtime, 8]))

diags.update(dict(PkrSPD4124 = [Float, 'W7X_MIR_4124', rawPeakSPD, 3]))
diags.update(dict(Pk4124 = [Float, 'W7X_MIR_4124', Peak, 3]))
diags.update(dict(Pkt4124 = [Float, 'W7X_MIR_4124', Peaktime, 3]))
diags.update(dict(Pkr4124 = [Float, 'W7X_MIR_4124', rawPeak, 3]))
diags.update(dict(Pkrt4124 = [Float, 'W7X_MIR_4124', rawPeaktime, 3]))

diags.update(dict(PkrSPD4136time = [Float, 'W7X_MIR_4136', rawPeakSPDtime, 8]))
diags.update(dict(PkrSPD4136 = [Float, 'W7X_MIR_4136', rawPeakSPD, 3]))
diags.update(dict(Pk4136 = [Float, 'W7X_MIR_4136', Peak, 3]))
diags.update(dict(Pkt4136 = [Float, 'W7X_MIR_4136', Peaktime, 3]))
diags.update(dict(Pkr4136 = [Float, 'W7X_MIR_4136', rawPeak, 3]))
diags.update(dict(Pkrt4136 = [Float, 'W7X_MIR_4136', rawPeaktime, 3]))

diags.update(dict(PkrSPD4114time = [Float, 'W7X_MIR_4114', rawPeakSPDtime, 8]))
diags.update(dict(PkrSPD4114 = [Float, 'W7X_MIR_4114', rawPeakSPD, 3]))
diags.update(dict(Pk4114 = [Float, 'W7X_MIR_4114', Peak, 3]))
diags.update(dict(Pkt4114 = [Float, 'W7X_MIR_4114', Peaktime, 4]))
diags.update(dict(Pkr4114 = [Float, 'W7X_MIR_4114', rawPeak, 3]))
diags.update(dict(Pkrt4114 = [Float, 'W7X_MIR_4114', rawPeaktime, 4]))

diags.update(dict(PkrSPD4132time = [Float, 'W7X_MIR_4132', rawPeakSPDtime, 8]))
diags.update(dict(PkrSPD4132 = [Float, 'W7X_MIR_4132', rawPeakSPD, 3]))
diags.update(dict(Pk4132 = [Float, 'W7X_MIR_4132', Peak, 3]))
diags.update(dict(Pkt4132 = [Float, 'W7X_MIR_4132', Peaktime, 4]))
diags.update(dict(Pkr4132 = [Float, 'W7X_MIR_4132', rawPeak, 3]))
diags.update(dict(Pkrt4132 = [Float, 'W7X_MIR_4132', rawPeaktime, 4]))

xdiags = dict(imain = [Float, 'need to define in pyfusion.cfg', Value],
             im2 = [Float, '.operations.magnetsupply.lcu.setup_main.i2', Value],
             im3 = [Float, '.operations.magnetsupply.lcu.setup_main.i3', Value],
             is1 = [Float, '.operations.magnetsupply.lcu.setup_sec.i1', Value],
             is2 = [Float, '.operations.magnetsupply.lcu.setup_sec.i2', Value],
             is3 = [Float, '.operations.magnetsupply.lcu.setup_sec.i3', Value],
             rf_drive = [Float, '.rf.rf_drive', Peak],
             i_fault = [Float, '.operations:i_fault', Average],
             rf_freq = [Float, '.log.heating.rf_freq', Value],
             llrf_mode = [Integer, '.log.heating.snmp.t1.operational.llrf.stallrfopm', Value],
             llrf_power = [Float, '.log.heating.snmp.t1.operational.llrf.stallrfppw', Value],
             llrf2_mode = [Integer, '.log.heating.snmp.t2.operational.llrf.stallrfopm', Value],
             llrf2_power = [Float, '.log.heating.snmp.t2.operational.llrf.stallrfppw', Value],
             v_main =  [Float, '.operations:v_main', Peak],
             int_v_main =  [Float, '.operations:v_main', Integral],
             v_sec =  [Float, '.operations:v_sec', Peak],
             int_v_sec =  [Float, '.operations:v_sec', Integral],
             i_sat_1 = [Float, '.fluctuations.BPP:vf_pin1',Std],
             i_sat_2 = [Float, '.fluctuations.BPP:vf_pin2',Std],
             isweep = [Float, '.fluctuations.BPP:isweep',Std],
             mirnov_coh =  [Float, '', Special],
         )
if pyfusion.NSAMPLES !=0:
    ans = input('Data secimation to {n} entries in force:  Continue?  (y/N)'.format(n=pyfusion.NSAMPLES,))
    if len(ans)==0 or ans.lower()[0] == 'n':
        sys.exit()
    input

# make the database infrastructure
col_list = [Column('shot', Integer, primary_key=True), 
            Column('date', Integer, primary_key=True),
            Column('sshot', Integer, primary_key=True),
            Column('took', Float),
            Column('length', Float),
            Column('dtsam', Float), #only approx
]

for diag in diags.keys():
    if len(diags[diag]) == 3: 
        typ, node, valfun = diags[diag]
        dp = None
    else:
        typ, node, valfun, dp = diags[diag]

    col_list.append(Column(diag, typ)) 

# create the table
summ = Table('summ', metadata, *col_list)

metadata.create_all(engine)
# simple method

ins = summ.insert()  # not sure why this is needed?

#import MDSplus as MDS
result = conn.execute('select count(*) from summ')
n = result.fetchone()[0]
if n > 0:
    print(conn.execute('select shot from summ').fetchall())
    ans = input('database {dbfile} is populated with {n} entries:  Continue?  (y/N/q=close)'.format(n=n, dbfile=dbfile))
    if len(ans)==0 or ans.lower()[0] == 'n':
        print("Example\n>>> conn.execute('select * from summ order by shot desc limit 1').fetchone()")
        if ans.lower()[0] == 'q':
            print('Closing db')
            conn.close()
        sys.exit()

have_shots=[longsht[0] for longsht in conn.execute('select shot from summ').fetchall()]
shots = 0

# set these both to () to stop on errors
shot_exception = () # Exception # () to see message - catches and displays all errors
node_exception = () if debug > 0 else Exception # ()  ditto for nodes.

errs = dict(shot=[])  # error list for the shot overall (i.e. if tree is not found)
for diag in diags.keys():
    errs.update({diag:[]})  # error list for each diagnostic

print("""If off-line, set pyfusion.TIMEOUT=0 to prevent long delays
The search for valid shot numbers is now much quicker
""")
# will take a while (~ 0.7 sec/day?),  about 2 mins as of 10/2017  

start = seconds()
dev = pyfusion.getDevice(devname)  # 'H1Local')
if 'W7' in devname:
    if (len(np.shape(srange)) == 3) or (np.shape(srange) == (2,2)):
        ansexp = input('About to expand shot range: Continue?  (Y/n/q)')
        if len(ans)==0 or ans.lower()[0] != 'y':
            sys.exit()
    
    pyfusion.logging.info(str('Starting with {st}, shape is {shp}'
                              .format(st=srangestr, shp=np.shape(srange))))
    if len(np.shape(srange)) == 3: # a bunch of begin/ends
        srange = [sh for sr in srange for sh in expand_shot_range(*sr)]
    elif np.shape(srange) == (2,2):
        print('assume a simple range')
        srange = expand_shot_range(*srange)
    else: # already a list
        pass 
else:
    srange=range(92000,95948)
print(srange)    
cache = 3*[None]
start_mem = report_mem(msg='Entry', prev_values = None)
# on t440p (83808,86450): # FY14-15 (86451,89155):#(36363,88891): #81399,81402):  #(81600,84084):
for (ish, sh) in enumerate(srange[::-1]):# may want [::-1] here to see the last first
    if 1000*sh[0] + sh[1] in have_shots:
        print('Skipping duplicate {shnum} '.format(shnum=str(sh))),
        continue

    cur_mem = report_mem(msg='next_shot', prev_values=start_mem, verbose=debug)
    # if we find the file 'pause' we pause, or if we find 'quit' we quit
    if pause_while(os.path.join(dbpath, 'pause'), check=2) == 'quit':
        break
    else:
        pass

    if 'W7' in devname:
        datdic = dict(shot=sh[0]*1000+sh[1], date=sh[0], sshot=sh[1])
    else:
        datdic = dict(shot=sh)

    if (ish % 10) == 0:
        print(sh, end=' ')
        if 1:  # the 'if' just is an artifice to indent  
            all_bad = []  # this simple logic gives false goods unless at end 
            for diag in errs.keys():
                all_bad.extend(errs[diag])
            bads = errs
            url = str(engine.url)
            shot_numbers = '-'.join([str(shnum).strip() for shnum in [np.sort(srange)[idx] for idx in [0, -1]]])
            goods = [s for s in srange if s not in all_bad] 
            save_path = 'tmp' if 'memory' in url else ''
            pfile = str('{spath}_{s}_{dt}_{fname}.log'
                        .format(dt=tm.strftime('%Y%m%d%H%M%S'), fname=os.path.split(url)[-1].replace(':','_'), spath=save_path,
                                s=shot_numbers.replace('(','').replace(')','').replace(']','_').replace('[','_')
                                .replace(',','_').replace(' ','')))

            print('See bads for {l} errors, also goods ({g}), and in {pfile}'
                  .format(l=len([b for b in bads.values() if len(b) > 0]), g=len(goods), pfile=pfile))
            json.dump(dict(bads=bads, goods=goods), open(pfile+'.json','w'))

    else: 
        sys.stderr.write('.' + str(sh))
    if (ish % 100) == 0: print('\n<<<{pc:.1f}% complete>>> '.format(pc=100*float(ish)/len(srange)), end='')

    shots += 1

    try:
        non_special = list(diags.keys())
        # non_special.remove('mirnov_coh')
        for diag in non_special:
            try:
                print(diag, end=', ')
                if len(diags[diag]) == 3: 
                    typ, node, valfun = diags[diag]
                    dp = None
                else:
                    typ, node, valfun, dp = diags[diag]

                if (cache[0] is not None) and (cache[0] == sh) and (cache[1] == node):
                    data = cache[2]
                else:
                    msg = str('>> no memory cache for {diag} on {sh}'
                              .format(sh=str(cache[0:2]), diag=diag))
                    print(msg, end=', ')
                    req_time = seconds()
                    data = dev.acq.getdata(sh, node)
                    if 'npz' not in data.params['source']:
                        pyfusion.logging.warn(msg.replace('memory', 'npz'))
                    took = seconds() - req_time
                    print('took ',took)
                    cache = [sh, node, data.copy()]
                    # hope to catch memory errors before they happen, but we still catch them is they do
                    mem_phys, mem_avail, mem_tot = report_mem(msg='after getdata', prev_values=None)
                    if mem_avail < 4e9:  # 12e9 is good for a test in the W7X VPCs 4e9 might be safe.
                        raise MemoryError('mem_avail = {mag:.2f}GB'.format(mag=mem_avail/1e9))
                if hasattr(data, 'timebase'):
                    tb = data.timebase
                    length = tb.max() - tb.min()
                    dtsam = np.diff(tb).mean()
                else:
                    tb = None
                val = valfun(data.signal, tb)
                if dp is not None:
                    val = round(val, dp)
                
                datdic.update({diag:val})
            except MemoryError as mem_reason:
                raise MemoryError('Shot {sh}, node {diag}, {mem_reason}'
                                  .format(sh=sh, diag=diag, mem_reason=mem_reason))
            except node_exception as reason:
                print('Node Exception: ', sh, node, reason.__repr__())
                errs[diag].append(list(sh))

        """         
        mtree=MDS.Tree('mirnov',sh)
        try:
            ch1 = mtree.getNode('.ACQ132_7.INPUT_01').data()
            ch2 = mtree.getNode('.ACQ132_7.INPUT_02').data()
            mdat = dict(#shot=sh, 
                mirnov_coh = np.max(np.abs(smooth_n(
                            ch1*ch2,500,iter=4))))
            datdic.update(mdat)
        except node_exception, reason:    
            errs['mirnov_coh'].append(sh)
        """
        datdic.update(dict(took=took)) # warning - these are for the last diag only
        datdic.update(dict(length=length))
    except MemoryError as mem_reason:
        pyfusion.logging.error(str('Memory Error shot {sh}, node {diag} {mr}'
                                   .format(sh=sh, diag=diag, mr=str(mem_reason))))
        break
    except shot_exception:
        errs['shot'].append(sh)
    else:  # executed if no exception
        conn.execute(ins, **datdic)

### Now the reading code 

from sqlalchemy import select
# note the .c attribute of the table object (column)
#if len(diags) > 10:
sel = select([summ.c.shot, eval('summ.c.'+diags.keys()[0])]) 
result = conn.execute(sel)  
row = result.fetchone()
#else
print(conn.execute('select * from summ limit 2').fetchall())

if row is None:
    raise LookupError('No data matching the query found - stopped by ./quit ?')
else:
    print('\nsample row = {r},\n  as items {it}\n  as dict {d}\n{b} bad/missing shots'
          .format(r=row, it=row.items(), d=dict(row), b=len(errs['shot'])))
all_bad = []
for diag in errs.keys():
    all_bad.extend(errs[diag])
    print('{0:11s} {1:6d} errors'.format(diag, len(errs[diag])))
print('{t} problems in {s} of {all} shots'.format(t=len(all_bad), s=len(np.unique(all_bad)),all=shots))

result.close()  # get rid of any leftover results
print('took {s:.1f} sec'.format(s=seconds()-start))

bads = errs
url = str(engine.url)
shot_numbers = '-'.join([str(shnum).strip() for shnum in [np.sort(srange)[idx] for idx in [0, -1]]])
goods = [s for s in srange if s not in all_bad] 
save_path = 'tmp' if 'memory' in url else ''
pfile = str('{spath}_{s}_{dt}_{fname}.log'
            .format(dt=tm.strftime('%Y%m%d%H%M%S'), fname=os.path.split(url)[-1].replace(':','_'), spath=save_path,
                    s=shot_numbers.replace('(','').replace(')','').replace(']','_').replace('[','_')
                    .replace(',','_').replace(' ','')))

print('See bads for {l} errors, also goods ({g}), and in {pfile}'
      .format(l=len([b for b in bads.values() if len(b) > 0]), g=len(goods), pfile=pfile))
json.dump(dict(bads=bads, goods=goods), open(pfile+'.json','w'))

"""
Examples:
result=conn.execute('select count(*) from summ where shot> 83808 and (isweep>2 or i_sat_1>0.2 or i_sat_2>0.2 or (mirnov_coh>0.02 and im2>2000))')

result=conn.execute('select shot,llrf_mode,rf_drive from summ where llrf_mode==1 and im2>6000')
result.fetchone()

result=conn.execute('select shot,rf_freq,llrf_mode,rf_drive from summ where llrf_mode==1 and im2>3000 and rf_freq > 5e6 order by rf_drive desc')


speed is about 1000 shots per minute (before I added time)

stats for Jul1 to Dec 31 2014
mirnov studies only (no probe sugs) 183
select count(*) from test.summ where shot> 83808 and not(i_sweep>2 or i_sat_1>0.2 or i_sat_2>0.2) and (mirnov_coh>0.02 and im2>2000); 

mirnov and probe 635
select count(*) from test.summ where shot> 83808 and (i_sweep>2 or i_sat_1>0.2 or i_sat_2>0.2 or (mirnov_coh>0.02 and im2>2000));


probe no mirnov 452
 select count(*) from test.summ where shot> 83808 and (i_sweep>2 or i_sat_1>0.2 or i_sat_2>0.2 or (mirnov_coh>10000000000.00 and im2>2000)); 

"""
