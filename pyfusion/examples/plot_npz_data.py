#!/usr/bin/env python
# this won't work from ~/bin unless we include pyfusion in PYTHONPATH
#import os
#print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import sys
sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.save_compress import newload

if len(sys.argv) < 2:
    print('plot_npz_data "filename"')
else:
    filename = sys.argv[1]
    dat = newload(filename)
    plt.plot(dat['timebase'], dat['signal'])
    plt.show(0)
