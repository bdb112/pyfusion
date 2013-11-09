""" timing on E4300  
Read a BLOSC h5 file containing pyfusion data: phases, shot, ne etc.
Advantage is very fast load of subranges (contiguous), and operation
directly on the data in files (out of core)

See also copy_DA and other code in ~/python/pytables_blosc/

12.4/10 sec for first/sec read of 20Mx16 f32 phases E4300 (zip=5)
after corruption? 51/18 sec for 1st/sec read of 20Mx16 f32 phases E4300 (zip=5)
Nov 2013 - 28/10 sec - takes 16-17 sec for npz file, and 35 sec to load whole file.
124/27 for 6X zip=1 - but much longer for [::10] (depends on swap)
efficient for slices in ranges, but "gather" is ~15x slower
 time pp=dd['phases'][unique(w/50),1:3]; print(shape(pp))

Good way to sample a data set is to take 10Million sample in the middle - 5 secs.
times can take 4-6x longer - is there an alignment problem?
* means relative to 4byte float.
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
