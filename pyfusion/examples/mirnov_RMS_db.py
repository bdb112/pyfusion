""" Create a table (RMS, and maybe later COH) with all mirnov channels
following mini_summary_MDS

Advantage of a separate table is:
 1/ if the main summary_h1 table is updated, this doesn't wipe the detailed mirnov data.
 2/ More compact as we can be selective
 3/ View lets us pretend they are in the same table - works in sqlite.9 on, seems as quick

>> drop view combsum;  -- sqlite3 example (maybe only v3.9 or higher)
>> create view combsum as select * from summary_h1, mirnov_RMS where mirnov_RMS.shot = summary_h1.shot;
A small down side is that using the combined db (view) excludes shots where mirnov_RMS 
  has not been calculated - could be solved by putting entries with nulls for the mirnov_RMS in all shots


Alternatives might be a blob or json representation in summary_h1, but then sql can't access mirnov_RMS data.

_PYFUSION_TEST_@@Skip
"""
import sys
import os
from time import time as seconds
from six.moves import input
import pyfusion
from pyfusion.data.shot_range import shot_range
from get_freq import mds_freq

from sqlalchemy import create_engine 
dbpath=''
sqlfilename = '/data/summary.db.new'
engine=create_engine('sqlite:///'+ os.path.join(dbpath, sqlfilename), echo=False)
#engine=create_engine('sqlite:///:memory:', echo=False)

conn = engine.connect()
import numpy as np


from sqlalchemy import Table, Date, Text, Column, Integer, String, Float, MetaData  # , ForeignKey

metadata = MetaData()

if len(sys.argv)>1:
    try:
        srange = eval(sys.argv[1])
    except:
        raise ValueError("Input was\n{inp}\nProbably need quotes:\n  Example:\n  run pyfusion/examples/mirnov_RMS_db 'range(88600,88732)  # shot_range is OK also'"
                         .format(inp=sys.argv))
else:
    srange = [88313, 88572, 88674]

# mcols =  {'H1ToroidalMirnov_3y': 'tm3y'}
mcol_list = [['H1ToroidalMirnov_{n}{X}'.format(n=n, X=X), 'm{n:02}{X}'.format(n=n, X=X)] for n in range(1, 17) for X in 'xyzt']
mcol_list.extend([['ElectronDensity_{n}'.format(n=n), 'ne{n:02}'.format(n=n)] for n in range(1, 21+1)])
mcol_list.extend([['accfreq', 'accf']])
mcols = {}
for mcol in mcol_list:  # the key is the long name, the value is the short name
    mcols.update({mcol[0]:mcol[1]})
    
col_list = [Column('shot', Integer, primary_key=True)]
for col in list(mcols):
    col_list.append(Column(mcols[col], Float))

# create the table
mirntab = Table('mirnov_RMS', metadata, *col_list)

metadata.create_all(engine)
# simple method

ins = mirntab.insert()  # convenience abbrev
# update is a different syntax - had trouble with UNIQUE constraint
#dele = mirntab.delete()  # there is no replace in sqlsoup?  so delete first


result=conn.execute('select count(*) from mirnov_RMS')
n = result.fetchone()[0]
if n > 0:
    ans = input('database is populated with {n} entries:  Continue?  (y/N)'.format(n=n))
    if len(ans)==0 or ans.lower()[0] == 'n':
        print("Example\n>>> conn.execute('select * from summ order by shot desc limit 1').fetchone()")
        sys.exit()

shots = 0

errs = dict(shot=[])  # error list for the shot overall (i.e. if tree is not found)
for mcol in mcol_list:
    errs.update({mcol[0]:[]})
    
start = seconds()

devname = 'H1Local' # 'W7X'
dev=pyfusion.getDevice(devname)  # 'H1Local'

for shot in srange:
    datdic = dict(shot=shot)
    do_later = [diag for diag in list(mcols) if mcols[diag].endswith('t')]
    do_now = [diag for diag in list(mcols) if not mcols[diag].endswith('t')]

    for diag in do_now:
        try:
            if diag == 'accfreq':
                datdic.update({mcols[diag]: float(mds_freq(shot=shot))})
            else:
                data = dev.acq.getdata(shot, diag)
                data = data.sp_filter_butterworth_bandpass(1e3, 2e3, 2, 20, btype='highpass')
                val = data.signal.std()
                datdic.update({mcols[diag]: float(val)})
        except:
            errs[diag].append(shot)
            
    for diag in do_later:  # t (theta) is the RMS of x and z cpts
        try:
            sqrs = [datdic[d]**2 for d in [mcols[diag.replace('t',X)] for X in 'xz']]
            datdic.update({mcols[diag]: float(np.sqrt(np.sum(sqrs)))})
        except:
            errs[diag].append(shot)
    
    conn.execute(mirntab.delete().where(mirntab.c.shot == shot))
    conn.execute(ins, **datdic)
    shots += 1

### Now the reading code 

from sqlalchemy import select
# note the .c attribute of the table object (column)
s = select([mirntab.c.shot, mirntab.c[mirntab.c.keys()[1]]]) 
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
