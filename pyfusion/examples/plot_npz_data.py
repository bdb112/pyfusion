#!/usr/bin/env python
# this won't work from ~/bin unless we include pyfusion in PYTHONPATH
#import os
#print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import sys
sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.save_compress import newload

# print(sys.argv)
if len(sys.argv) < 2:
    print('plot_npz_data "filename"')
else:
    filename = sys.argv[1]
    dat = newload(filename)
    plt.plot(dat['timebase'], dat['signal'])
    # hold if called from a bin directory - otherwise (e.g. interactive) don't
    # originally tested if '/bin' in argv[0], but this fails for run ~/bin/plot.. under ipython
    block_me = not hasattr(sys, 'ps1')
    print('block_me = {b}, args={a}'.format(b=block_me, a=sys.argv))
    if 'params' in list(dat):
        import pprint
        pprint.pprint(dat['params'], indent=4)

    plt.show(block=block_me)
