"""  to be adapted to pyfusion from  a version hard-coded for H-1

General pyfusion version of mini_summary - see mini_summary_MDS for MDSPlus specific version

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
import os
import sys
from pyfusion.data.signal_processing import smooth_n
from time import time as seconds
import pyfusion
from six.moves import input

from sqlalchemy import create_engine 
#engine=create_engine('sqlite:///:memory:', echo=False)
# put the file in the example (this) folder although it may be better in the acquisition/W7X folder

devname = 'W7X'
if 'W7' in devname:
    sqlfilename = '/W7X_mag_OP1_1.sqlite'
else:
    sqlfilename = '/H1.sqlite'

dbpath = os.path.dirname(__file__)
engine=create_engine('sqlite:///'+ dbpath + sqlfilename, echo=False)
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
    print 'dumping to ' + fnamefull
    pickle.dump(list,file(fnamefull,'w'))

###########################
# Functions to process data
def Peak(x,t):
    """ 98th percentile """
    return np.sort(x)[int(0.98*len(x))]

def Special():
    # hard coded for now
    pass


def Std(x, t):
    return(np.std(x))

def Average(x,t=None):
# should really to an average over an interval
    if t is None: t = x.timebase
    w = np.where((t>interval[0]) & (t>interval[1]))[0]
    return np.average(x.signal[w])

def Integral(x,t):
# should really include dt
    return np.average(np.diff(t))*np.sum(x)

def Value(x,t):
    if np.issubdtype(type(x), int):
        return(int(x))
    else:
        return(x)

metadata = MetaData()

if len(sys.argv)>1:
    try:
        srange = eval(sys.argv[1])
    except:
        raise ValueError("Input was\n{inp}\nProbably need quotes:\n  Example:\n  run pyfusion/examples/mini_summary 'range(88600,88732)'"
                         .format(inp=sys.argv))
else:
    srange = range(88600,88732)


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
diags = dict(IPlanar_A = [Float, 'W7X_IPlanar_A', Average, 1],
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
             # partial, and needs more care peak or average (spikes?)
             #ECH_Rf_A1 = [Float, 'W7X_ECH_Rf_A1', Average, 1],
             #ECH_Rf_B1 = [Float, 'W7X_ECH_Rf_B1', Average, 1],
             #ECH_Rf_C1 = [Float, 'W7X_ECH_Rf_C1', Average, 1],
             #ECH_Rf_D1 = [Float, 'W7X_ECH_Rf_D1', Average, 1],
             #ECH_Rf_A5 = [Float, 'W7X_ECH_Rf_A5', Average, 1],
             #ECH_Rf_B5 = [Float, 'W7X_ECH_Rf_B5', Average, 1],
             #ECH_Rf_C5 = [Float, 'W7X_ECH_Rf_C5', Average, 1],
             #ECH_Rf_D5 = [Float, 'W7X_ECH_Rf_D5', Average, 1],
             
             imain = [Float, 'W7X_INonPlanar_1', Average, 1],)

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

# make the database infrastructure
col_list = [Column('shot', Integer, primary_key=True), 
            Column('date', Integer, primary_key=True),
            Column('sshot', Integer, primary_key=True)]

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

ins=summ.insert()  # not sure why this is needed?

#import MDSplus as MDS
result=conn.execute('select count(*) from summ')
n = result.fetchone()[0]
if n> 0:
    ans = input('database is populated with {n} entries:  Continue?  (y/N)'.format(n=n))
    if len(ans)==0 or ans.lower()[0] == 'n':
        print("Example\n>>> conn.execute('select * from summ order by shot desc limit 1').fetchone()")
        sys.exit()
shots = 0

# set these both to () to stop on errors
shot_exception = () # Exception # () to see message - catches and displays all errors
node_exception = () # Exception # ()  ditto for nodes.

errs = dict(shot=[])  # error list for the shot overall (i.e. if tree is not found)
for diag in diags.keys():
    errs.update({diag:[]})  # error list for each diagnostic

print("""If off-line, set pyfusion.TIMEOUT=0 to prevent long delays
The search for valid shot numbers will take a while.  about 2 mins as of 10/2017  
""")


start = seconds()
dev=pyfusion.getDevice(devname)  # 'H1Local')
from pyfusion.data.shot_range import shot_range as expand_shot_range
srange = ((20160309,1), (20160310,99))
srange = ((20160101,1), (20160310,99))
srange = ((20160101,1), (20171110,99))
#srange = ((20160202,1), (20160202,99))
if 'W7' in devname:
    srange = expand_shot_range(*srange)
else:
    srange=range(92000,95948)

for sh in srange: # on t440p (83808,86450): # FY14-15 (86451,89155):#(36363,88891): #81399,81402):  #(81600,84084):
    if 'W7' in devname:
        datdic = dict(shot=sh[0]*1000+sh[1], date=sh[0], sshot=sh[1])
        if (sh[1] % 10) == 0:
            print(sh),
        else: 
            sys.stderr.write('.')
        if (sh[1]%100)==0: print('')
    else:
        datdic = dict(shot=sh)
        if (sh%10)==0: 
            sys.stderr.write(str(sh))
        else: 
            sys.stderr.write('.')

    shots += 1
    try:
        non_special = list(diags.keys())
        # non_special.remove('mirnov_coh')
        for diag in non_special:
            try:
                if len(diags[diag]) == 3: 
                    typ, node, valfun = diags[diag]
                    dp = None
                else:
                    typ, node, valfun, dp = diags[diag]

                data = dev.acq.getdata(sh, node)
                if hasattr(data, 'timebase'):
                    tb = data.timebase
                else:
                    tb = None
                val = valfun(data)
                if dp is not None:
                    val = round(val, dp)
                
                datdic.update({diag:val})
            except Exception, reason:
                #print sh,node,reason
                errs[diag].append(sh)

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
    except shot_exception:
        errs['shot'].append(sh)
    else:  # executed if no exception
        conn.execute(ins, **datdic)

### Now the reading code 

from sqlalchemy import select
# note the .c attribute of the table object (column)
sel = select([summ.c.shot, summ.c.imain]) 
result = conn.execute(sel)  
row = result.fetchone()
print('\nsample row = {r},\n  as items {it}\n  as dict {d}\n{b} bad/missing shots'
      .format(r=row, it=row.items(), d=dict(row), b=len(errs['shot'])))
all_bad = []
for diag in errs.keys():
    all_bad.extend(errs[diag])
    print '{0:11s} {1:6d} errors'.format(diag, len(errs[diag]))
print('{t} problems in {s} of {all} shots'.format(t=len(all_bad), s=len(np.unique(all_bad)),all=shots))

result.close()  # get rid of any leftover results
print('took {s:.1f} sec'.format(s=seconds()-start))

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
