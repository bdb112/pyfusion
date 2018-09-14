"""  to be adapted to pyfusion from  a version hard-coded for H-1

MDSPlus specific version of mini_summary - see mini_summary for general pyfusion version

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

Advanced: plotting
sh,k_h = np.transpose(conn.execute('select shot, is2/im2 as kh from summ where im2>2000').fetchall())
# one liner
plot(*(np.transpose(conn.execute('select shot, is2/im2 as kh from summ where im2>2000').fetchall()).tolist()),marker=',',linestyle=' ') 
scatter(*(np.transpose(conn.execute('select shot, is2/im2 as kh from summ where im2>2000').fetchall()).tolist()),marker='.')  # , not known to scatter


duds 36362
"""
import sys
import os
from pyfusion.data.signal_processing import smooth_n
from time import time as seconds
from six.moves import input

from sqlalchemy import create_engine 
sqlfilename = 'MP2.sqlite'
dbpath = os.path.dirname(__file__)
engine = create_engine('sqlite:///' + os.path.join(dbpath, sqlfilename), echo=False)
#engine=create_engine('sqlite:///:memory:', echo=False)
#engine=create_engine('sqlite:///testmds.sqlite', echo=False)
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

from sqlalchemy import DateTime, Table, Column, Integer, String, Float, MetaData#, ForeignKey


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

"""
https://groups.google.com/forum/#!topic/comp.os.vms/0bCcHTumgl8
See python/VMS_time.py
"""
def VMS_time(i64):
   from datetime import datetime,timedelta
   us = int(i64) / 10.
   print datetime(1858,11,17) + timedelta(microseconds=us)
   
###########################
# Functions to process data
def Peak(x,t):
    return np.sort(x)[int(0.98*len(x))]

def Special():
    # hard coded for now
    pass


def Std(x, t):
    # This is like AC RMS
    return(np.std(x))

def RMS(x,t):
# like DC_RMS
    w = np.where((t>interval[0]) & (t>interval[1]))[0]
    return plt.mlab.rmsflat(x[w])

def Average(x,t):
# should really to an average over an interval
    w = np.where((t>interval[0]) & (t>interval[1]))[0]
    return np.average(x[w])

def Integral(x,t):
# should really include dt
    return np.average(np.diff(t))*np.sum(x)

def Value(x,t):
    if np.issubdtype(type(x), int):
        return(int(x))
    elif isinstance(x, (str)):
        return(x[0])
    else:
        return(x)

def StringValue(x,t):
    if isinstance(x, (np.ndarray)):
        return(x[0])
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
    srange = range(11668, 11961+1)


diags = dict(comment = [String, '.log.machine.comment', StringValue],
             timeInserted = [DateTime, '', Special],  # do separately
             target_amps = [Float, '.MAGNETICS:TARGET_AMPS', Value],
             source_amps = [Float, '.MAGNETICS:SOURCE_AMPS', Value],
             freq = [Float, '.RF:FREQ', Value],
             p_net = [Float, '.RF:P_NET', Average],
             p_net_peak = [Float, '.RF:P_NET', Peak],
             img_rank = [Integer,'SHAPE(SHAPE(.CAMERA:IMAGE), 0)', Value],
             img_dim0 = [Integer,'SHAPE(.CAMERA:IMAGE, 0)', Value],
             img_dim1 = [Integer,'SHAPE(.CAMERA:IMAGE, 1)', Value],
             img_dim2 = [Integer,'SHAPE(.CAMERA:IMAGE, 2)', Value],
             rf_drive = [Float, '.RF.RF_DRIVE', Peak],
             probe_i = [Float, '.PROBE:I_SWEEP', Average],
             probe_di = [Float, '.PROBE:I_SWEEP', Std],
             probe_v = [Float, '.PROBE:V_SWEEP', Average],
             probe_dv = [Float, '.PROBE:V_SWEEP',Std],
         )
col_list = [Column('shot', Integer, primary_key=True)]
for diag in diags.keys():
    typ, node, valfun = diags[diag]
    col_list.append(Column(diag, typ)) 

# create the table
summ = Table('summ', metadata, *col_list)

metadata.create_all(engine)
# simple method

ins=summ.insert()  # not sure why this is needed?
result=conn.execute('select count(*) from summ')
n = result.fetchone()[0]
if n> 0:
    ans = input('database is populated with {n} entries:  Continue?  (y/N)'.format(n=n))
    if len(ans)==0 or ans.lower()[0] == 'n':
        print("Example\n>>> conn.execute('select * from summ order by shot desc limit 1').fetchone()")
        sys.exit()

import MDSplus as MDS

shots = 0

# set these both to () to stop on errors
shot_exception = ()  # () to see message - catches and displays all errors
node_exception =  Exception  # ()  ditto for nodes.

errs = dict(shot=[])  # error list for the shot overall (i.e. if tree is not found)
for diag in diags.keys():
    errs.update({diag:[]})  # error list for each diagnostic

start = seconds()
for s in srange: # on t440p (83808,86450): # FY14-15 (86451,89155):#(36363,88891): #81399,81402):  #(81600,84084):
    datdic = dict(shot=s)
    shots += 1
    try:
        if (s%10)==0: 
            print(s),
        else: 
            print('.'),
        if (s%100)==0: print('')
        tree=MDS.Tree('magpie2',s)
        non_special = [diag for diag in list(diags) if diags[diag][2] != Special] # if 'dim' in diag]
        #non_special.remove('mirnov_coh')
        for diag in non_special:
            try:
                typ, node, valfun = diags[diag]
                # nd = tree.getNode(node)
                nd = MDS.Data.compile(node)  #  to allow MDS expressions
                if 'dimxxx0' in diag: #str(nd):
                    1/0
                try:
                    if valfun != Value:  # no dim expected here.
                        dim = nd.dim_of().data()
                    else:
                        dim = None
                    val = valfun(nd.data(), dim)
                except (MDS.TreeNNF, MDS.TreeNODATA, MDS.TdiBAD_INDEX, MDS.TdiINVCLADSC, MDS.TreeBADRECORD) as reason:
                    dim = None
                    val = None
                    raise reason
                finally:
                    datdic.update({diag:val})
            except node_exception, reason:
                #print s,node,reason
                errs[diag].append((s, reason))

        try:
            #mtree=MDS.Tree('mirnov',s)
            #ch1 = mtree.getNode('.ACQ132_7.INPUT_01').data()
            #ch2 = mtree.getNode('.ACQ132_7.INPUT_02').data()
            #mdat = dict(#shot=s, 
            #    mirnov_coh = np.max(np.abs(smooth_n(
            #                ch1*ch2,500,iter=4))))
            #datdic.update(mdat)
            pass
        except node_exception, reason:    
            pass
            # errs['special'].append(s)

    except shot_exception:
        errs['shot'].append(s)
    else:  # executed if no exception
        conn.execute(ins, **datdic)

### Now the reading code 

from sqlalchemy import select
# note the .c attribute of the table object (column)
s = 'select * from summ' 
result = conn.execute(s)  
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

# make a DA file from the whole database
result = conn.execute('select * from summ order by shot' ) # where shot>90000')  # 90000
# result = conn.execute("select * from summ where proc_time> date('2016-12-20')")
xx = result.fetchall()
xxt = np.array(xx).T
# put them in a dictionary
# note that the presence of Nones requires fiddling to avoid 'objects'
dat = {}
for (k, key) in enumerate(xx[0].keys()):
    vals =  xxt[k]
    bads = [v is None for v in vals]
    vals[[np.where(bads)[0]]] = np.nan
    dat.update({key: np.array(vals.tolist())})

# put the dictionary in a DA
from pyfusion.data.DA_datamining import Masked_DA, DA
dam = DA(dat)
# make each one an attribute
# nice trick, but won't survive a save unless we explicitly re-do this on restore.
for key in dam:
    if not hasattr(dam, key):
        exec("dam."+key+"=dam['" + key + "']")


# a range of shots suitable for testing
from pyfusion.data.convenience import inlist, between
dam.extract(locals())  # all shots

"""
