#!/usr/bin/env python
# python3 error in save_compress.py", line 397
import os
#print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import sys
PF_path = os.getenv('PYFUSION_PATH','/home/bdb112/pyfusion/working/pyfusion/')
print('assume pyfusion is at ', PF_path, ':  can override by setting PYFUSION_PATH') 
sys.path.append(PF_path)
from pyfusion.data.save_compress import newload

# print(sys.argv)
if len(sys.argv) < 2:
    print('plot_npz_data "filename"')
else:
    filename = sys.argv.pop(1)
    verbose = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    dat = newload(filename, verbose=abs(verbose))  # -1 means info only no plot
    if verbose >= 0:
        plt.plot(dat['timebase'], dat['signal'])
        # hold if called from a bin directory - otherwise (e.g. interactive) don't
        # originally tested if '/bin' in argv[0], but this fails for run ~/bin/plot.. under ipython
    block_me = not hasattr(sys, 'ps1')

    print('block_me = {b}, args={a}'.format(b=block_me, a=sys.argv))
    if 'params' in list(dat):
        import pprint
        pprint.pprint(dat['params'], indent=4)

    try:
        print('timebase is {0}, signal is {1}'
              .format(dat['timebasetype'], dat['signaltype']))
    except:
        print('Error looking at data types - maybe need pyfusion.VERBOSE > 0 or plot_npz_data <file> 1 ')
    if verbose >= 0:
        plt.show(block=block_me)
    
