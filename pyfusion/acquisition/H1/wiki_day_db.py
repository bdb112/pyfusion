""" Use to fill a wikiday table in summary_h1 using files saved by scrape_wiki

run -i pyfusion/acquisition/H1/scrape_wiki.py http://h1svr.anu.edu.au/wiki/Day/2016
run -i pyfusion/acquisition/H1/scrape_wiki.py http://h1svr.anu.edu.au/wiki/Day/2017


run -i pyfusion/acquisition/H1/wiki_day_db.py

following mini_summary_MDS

"""
import sys
import os
import time as tm
from time import time as seconds
from six.moves import input

from sqlalchemy import create_engine 
dbpath=''
sqlfilename = '/data/summary.db'
engine=create_engine('sqlite:///'+ os.path.join(dbpath, sqlfilename), echo=False)
dbpath = os.path.dirname(__file__)
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

from sqlalchemy import Table, Date, Text, Column, Integer, String, Float, MetaData#, ForeignKey

############
## Utilities

metadata = MetaData()

if len(sys.argv)>1:
    try:
        yrange = eval(sys.argv[1])
    except:
        raise ValueError("Input was\n{inp}\nProbably need quotes:\n  Example:\n  run pyfusion/examples/mini_summary 'range(88600,88732)'"
                         .format(inp=sys.argv))
else:
    yrange = range(2009,2020)


col_list = [Column('wikiday', Date, primary_key=True),Column('topic', Text)]

# create the table
wikidays = Table('wikidays', metadata, *col_list)

metadata.create_all(engine)
# simple method

ins = wikidays.insert()  # ins is a shortcut for insert
result = conn.execute('select count(*) from wikidays')
n = result.fetchone()[0]
if n > 0:
    ans = input('database is populated with {n} entries:  Continue?  (y/N)'.format(n=n))
    if len(ans)==0 or ans.lower()[0] == 'n':
        print("Example\n>>> conn.execute('select * from summ order by shot desc limit 1').fetchone()")
        sys.exit()

days = 0

errs = dict(day=[])  # error list for the shot overall (i.e. if tree is not found)

start = seconds()
import json
for y in yrange: 
    pagedict = json.load(open('wikipagedict_{y}.json'.format(y=y)))
    for k in pagedict:
        days += 1
        tst = tm.strptime(k.split('Day/')[-1], '%Y/%m/%d')
        datdic = dict(wikiday = datetime.date(tst.tm_year, tst.tm_mon, tst.tm_mday), topic = pagedict[k])
        conn.execute(ins, **datdic)

### Now the reading code 

from sqlalchemy import select
# note the .c attribute of the table object (column)
s = select([wikidays.c.wikiday, wikidays.c.topic]) 
result = conn.execute(s)  
row = result.fetchone()
print('\nsample row = {r},\n  as items {it}\n  as dict {d}\n{b} bad/missing shots'
      .format(r=row, it=row.items(), d=dict(row), b=len(errs['day'])))
all_bad = []
for diag in errs.keys():
    all_bad.extend(errs[diag])
    print '{0:11s} {1:6d} errors'.format(diag, len(errs[diag]))
print('{t} problems in {s} of {all} days'.format(t=len(all_bad), s=len(np.unique(all_bad)),all=days))

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
