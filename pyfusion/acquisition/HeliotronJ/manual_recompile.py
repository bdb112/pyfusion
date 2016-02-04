# force a recompile for test purposes, and run a simple test
# only works after a restart of ipython
import os
import sys
import numpy as np

from pyfusion.acquisition.HeliotronJ import get_hj_modules
mod_name = get_hj_modules.get_hj_modules()
# .so only works in linux
full_name = os.path.join(os.path.split(mod_name[1])[0],mod_name[0]+'.so')
os.unlink(full_name)
mod_name = get_hj_modules.get_hj_modules()

from pyfusion.acquisition.HeliotronJ import gethjdata2_7
x = np.zeros([1e6])
y = x*1

try:
    shot = int(sys.argv[1])
except:
    shot = 61818

#gethjdata2_7.gethjdata(58000,100,'DIA135',verbose=1,opt=0,outname='foo',outdata=x)
# this is one file I have set up on my computer under /data/160202/61818/MP1
# and /data/HDISK.lst and HDISK have been copied from HeliotronJ
# note that the original code requires that HDISK.list be in fixed format.
#  at the moment, this gives no errors but returns all zeros?
gethjdata2_7.gethjdata(shot,100,'MP1',verbose=1,opt=0,outname='foo',outdata=x)
if x[1] == y[1]: print('no new data returned in x??')
