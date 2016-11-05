#!/usr/bin/env python
# this won't work from ~/bin unless we include pyfusion in PYTHONPATH
#import os
#print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import sys
sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.DA_datamining import DA 

if len(sys.argv) < 2:
    print('plot_npz_data "filename" key')
else:
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        key = sys.argv[2]
    else:
        key = 'ne18'

    da = DA(filename,load=1)
    da.plot(key)
    plt.show(1)
