#!/usr/bin/env python
"""
Show the info dict of a DA_datamining npz file
 This won't work from ~/bin unless we include pyfusion in PYTHONPATH - do this with a crude hack below
import os
print(os.path.dirname(__file__))
_PYFUSION_TEST_@@pyfusion/examples/JSPF_tutorial/H1_766.npz 
_PYFUSION_TEST_@@pyfusion/examples/JSPF_tutorial/LP20160309_52_L53_2k2short.npz
"""


import matplotlib.pyplot as plt
import sys
sys.path.append('/home/bdb112/pyfusion/working/pyfusion/')
from pyfusion.data.DA_datamining import DA 

if len(sys.argv) < 2:
    raise Exception('Syntax is:  DA_info "filename"')
else:
    filename = sys.argv[1]

from pyfusion.data.DA_datamining import Masked_DA, DA
da = DA(filename)
print(da['info'])
