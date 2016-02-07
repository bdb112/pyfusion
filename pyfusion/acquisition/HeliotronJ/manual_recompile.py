# force a recompile for test purposes, and run a simple test
# only works after a restart of ipython
import os
import sys
import numpy as np

from pyfusion.acquisition.HeliotronJ.get_hj_modules import get_hj_modules, import_module
# Note:  don't ask for the module name yet, make new file first!
get_hj_modules(force_recompile=True)

hjmod, exe_path = get_hj_modules()
import_module(hjmod,'gethjdata',locals())

x = np.zeros(int(1e6))
y = x*1

try:
    shot = int(sys.argv[1])
except:
    shot = 61818

#gethjdata.gethjdata(58000,100,'DIA135',verbose=1,opt=0,ierror=1,outname='foo',outdata=x)
# this is one file I have set up on my computer under /data/160202/61818/MP1
# and /data/HDISK.lst and HDISK have been copied from HeliotronJ
# note that the original code requires that HDISK.list be in fixed format.
#  at the moment, this gives no errors but returns all zeros?
ierr,retdata = gethjdata.gethjdata(shot,100,'MP1',verbose=1,opt=1,ierror=1,outname='foo',outdata=x)
if ierr != 0: print('ierror = ',ierr)
if x[1] == retdata[1]: print('no new data returned in x??')

