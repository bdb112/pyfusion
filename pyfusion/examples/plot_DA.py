#!/usr/bin/env python
from __future__ import print_function

# this won't work from ~/bin unless we include pyfusion in PYTHONPATH
# import os
# print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import sys
import os
import pprint

PF_path = os.getenv('PYFUSION_PATH','/home/bdb112/pyfusion/working/pyfusion/')
print('assume pyfusion is at ', PF_path, ':  can override by setting PYFUSION_PATH') 
sys.path.append(PF_path)
sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.DA_datamining import DA 

if len(sys.argv) < 2:
    print('plot_npz_data "filename" key')
else:
    filename = sys.argv[1]
    if os.path.split(filename)[1] == filename:
        print(os.getcwd(), end='')
    print(filename)

    # from_emacs was called block_me
    from_emacs = not hasattr(sys, 'ps1')
    if from_emacs:
        print('from emacs')
        os.environ['PYFUSION_VERBOSE'] = '-1'  # keep clutter down

    import pyfusion
    pyfusion.VERBOSE = -1   # why doesn't the environ above work?
    da = DA(filename, load=1, verbose=pyfusion.VERBOSE)
    if 'info' in da:
        pprint.pprint(da['info'])
    if hasattr(da, 'info'):
        da.info()
    sys.stdout.flush()  # works, but not effective in emacs, at least when ! is used.
    
    if len(sys.argv) > 2:
        key = sys.argv[2]
    elif 'ne18' in da:
        typ = 'LP'
        key = 'ne18'
    elif 'phases' in da:
        typ = 'FS'
        key = 'phases'
    else:
        raise LookupError('Keys available are: ' + str(list(da)))

    da.plot(key)
    plt.show(1)
