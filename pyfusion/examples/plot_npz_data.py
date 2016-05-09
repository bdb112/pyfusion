#!/usr/bin/env python
# this won't work from ~/bin unless we for PYTHONPATH
#import os
#print(os.path.dirname(__file__))

from pyfusion.data.save_compress import newload
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print('plot_npz_data "filename"')
else:
    filename = sys.argv[1]
    dat = newload(filename)
    plt.plot(dat['timebase'], dat['signal'])
    plt.show(0)
