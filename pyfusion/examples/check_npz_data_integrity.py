""" Written to check for comparison with maxint after scaling, but can be hacked
    for other purposes
_PYFUSION_TEST_@@filepath='/data/datamining/local_data/W7X/*npz'
"""
from __future__ import print_function
import glob2
import numpy as np
from pyfusion.utils.process_cmd_line_args import process_cmd_line_args

filepath = '/media/bdb1)12/BIG360/data/**/*.npz'

exec(process_cmd_line_args())

files = glob2.glob(filepath)

for (c, fn) in enumerate(files):
    dic = np.load(fn)
    if 'timebaseexpr' not in list(dic):
        print('No timebaseexpr in ' + fn)
        continue
    timebaseexpr = dic['timebaseexpr'].tolist()
    if 'maxint == dic' not in timebaseexpr and 'maxint' in timebaseexpr:
        print(f,dic['timebaseexpr'].tolist())
    if c%100 == 0:
        print('.', end = ['', '\n'][((c+1)%10000 == 0)] )
