#!/usr/bin/env python
from __future__ import print_function

# this won't work from ~/bin unless we include pyfusion in PYTHONPATH
# import os
# print(os.path.dirname(__file__))


import matplotlib.pyplot as plt
import sys
import os
import pprint

sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.DA_datamining import DA 

if len(sys.argv) < 2:
    print('plot_npz_data "filename" key')
else:
    filename = sys.argv[1]
    if os.path.split(filename)[1] == filename:
        print(os.getcwd(), end='')
    print(filename)
    if len(sys.argv) > 2:
        key = sys.argv[2]
    else:
        key = 'ne18'

    da = DA(filename, load=1)
    da.plot(key)
    if 'info' in da:
        pprint.pprint(da['info'])
    if hasattr(da, 'info'):
        da.info()
    plt.show(1)
