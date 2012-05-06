"""  run pyfusion/examples/check_other_data.py shots="np.loadtxt('lhd_clk1.txt',dtype=type(1))"
"""

import pyfusion as pf
import pylab as pl
import numpy as np
from read_igetfile import igetfile

verbose = 0
shots=np.loadtxt('lhd_clk4.txt',dtype=type(1))
fileformat = 'cache/fircall@{0}.dat.bz2'
separate=0

import pyfusion.utils
exec(pf.utils.process_cmd_line_args())

missing_shots = []
good_shots =[]
for shot in shots:
    try:
        dat=igetfile(fileformat,shot=shot)
    	if separate: pl.figure()

        dat.plot(hold=1,ch=1)
        good_shots.append(shot)
    except IOError:		
        missing_shots.append(shot)

pl.show()

print("{0} missing shots out of {1}".format(len(missing_shots),(len(missing_shots)+len(good_shots))))

if verbose>0: print('missing shots are {0}'.format(missing_shots))
