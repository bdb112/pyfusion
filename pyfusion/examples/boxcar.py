from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import sys

#  run pyfusion/examples/plot_signals.py dev_name=W7X diag_name=W7X_LTDU_LP09_U shot_number=[20180821,18] stop=0
# this run -i

period = 1001.89   # actual period in samples
ip = int(period)  # integer part of the period
numcyc = len(v) // ip - 2   # take off 2 for a safety margin
numcyc = numcyc - numcyc//ip # and allow for 1 extra point lost each cycle
numcyc = min(10000, numcyc)  # can reduce number as we do the tuning.

ar2d = []
for cyc in range(numcyc):
    offs = int(cyc * (period - ip))  # offset crows a little each time
    ar2d.append(v[cyc * ip + offs: (cyc + 1) * ip + offs])

plt.plot(np.mean(ar2d,axis=0))

    
