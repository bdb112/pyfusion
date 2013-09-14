""" timing on E4300  

12.4/10 sec for first/sec read of 20Mx16 f32 phases E4300 (zip=5)
after corruption? 51/18 sec for 1st/sec read of 20Mx16 f32 phases E4300 (zip=5)
124 for 6X zip=1 - but much longer for [::10]
efficient for slices in ranges, but "gather" is ~15x slower
 time pp=dd['phases'][unique(w/50),1:3]; print(shape(pp))

* means relative to 4bye float.
"""
import tables as tb
import os
import numpy as np
from bdb_utils import process_cmd_line_args
from time import time as seconds
from pyfusion.data.DA_datamining import DA, report_mem

_var_defaults = """
DFfilename='/data/datamining/PF2_130813_50_5X_1.5_5b_rms_1_diags_comp5.h5'
keep_open = 1  # for read, should keep open!
debug=1
"""
exec(_var_defaults)
exec(process_cmd_line_args())

df = tb.openFile(DFfilename,'r')
dd = {}

for nd in df.listNodes('/'):
    var = nd.name
    st_copy = seconds()

    v = df.getNode('/'+var)
    dd.update({var:v})
    
    dt_copy = seconds() - st_copy
    if debug>0:
        print('{var} in {dt_copy:.1f}s '.format(var=var, dt_copy=dt_copy))

if (not keep_open):
    df.close()
