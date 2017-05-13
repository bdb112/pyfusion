#!/usr/bin/env python
"""
Show the info dict of a DA_datamining npz file
 This won't work from ~/bin unless we include pyfusion in PYTHONPATH - do this with a crude hack below
import os
print(os.path.dirname(__file__))
"""


import matplotlib.pyplot as plt
import sys
sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.DA_datamining import DA 

if len(sys.argv) < 2:
    print('DA_info "filename"')
else:
    filename = sys.argv[1]

from pyfusion.data.DA_datamining import Masked_DA, DA
da = DA(filename)
print da['info']
