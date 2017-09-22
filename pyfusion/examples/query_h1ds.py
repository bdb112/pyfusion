""" Creates a dictionary from the H1 database (modified from Dave's production 
Can be saved by DA_datamining.
"""
# Need to add t1,t2,t3, i_hel, densities

import pylab as pl
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import or_, and_, desc, asc
import time as tm

try:
    from sqlsoup import SQLSoup
except ImportError:
    print('trying to import obsolete form oof sqlsoup')
    from sqlalchemy.ext.sqlsoup import SqlSoup as SQLSoup


def replace_none(arr, val=np.nan):
    outarr = []
    for elt in arr:
        if elt is None:
            outarr.append(val)
        else:
            outarr.append(elt)
    return(np.array(outarr))

#engine = create_engine('mysql://localhost/from_h1ds')
engine=create_engine('sqlite:////home/bdb112/data/datamining/h1ds_production_2013_09_15.db')
db=SQLSoup(engine)
#db.summary.shot.between(78000,78010)
q=db.summary.filter(db.summary.shot.between(000,80000))
#q=db.summary.filter(db.summary.shot.between(79000,79010))a
#q=db.summary.filter(db.summary.shot.between(79300,79350))
allres=q.all()
summary = {}
for k in "shot,kappa_h,rf_drive,timestamp,rfp_req,im1,im2,im3,is1,is2,is3,sec_t1,sec_t2,i_main,i_ring,i_sec,"\
        "rf1_cool,rf2_cool,rf3_cool,rf4_cool,PHDIFF_COOL,rf_freq_mhz,"\
        "ne18_bar,lcu_gas_1_flow,lcu_gas_2_flow,lcu_gas_3_flow".split(','):
    dat = [eval('r.'+k) for r in allres]
    summary.update({k:np.array(dat)})
    
# probably should do this for all.
for k in "im1,im2,im3,is1,is2,is3,kappa_h,rfp_req".split(','):
    summary.update({k:replace_none(summary[k], np.nan)})

summary.update({'ymd':np.array([d.date() for d in summary['timestamp']])})

fig, axs = pl.subplots(4,1,sharex='all')
axs[0].plot(summary['kappa_h'],'+-', ms=3, label='k_h')
axs[0].legend()
axs[1].plot(summary['im1']/13888.,',',ms=1);axs[1].set_ylabel('Bo')
axs[2].plot(summary['rf_drive'],',',ms=1);axs[2].set_ylabel('rf_drive')
axs[3].plot(summary['shot'])
#axs[3].set_ylim(70000,80000)

pl.show()
