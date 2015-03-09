""" Plot the svd of a diag
"""

import pyfusion as pf
import pyfusion.utils
import pylab as pl
import numpy as np


diag_name = ''
dev_name='H1Local'   # 'LHD'
hold=0
exception=Exception
time_range = None
channel_number=0
start_time = None
numpts = 512
normalise='0'
help=0
separate=1
verbose=0
max_fs = 2
shot_number = None
i=1

#execfile('process_cmd_line_args.py')
exec(pf.utils.process_cmd_line_args())
if help==1: 
    print(__doc__) 
    exit()

#dev_name='LHD'
if dev_name == 'LHD': 
    if diag_name == '': diag_name= 'MP2010'
    if shot_number is None: shot_number = 27233
    #shot_range = range(90090, 90110)
elif dev_name.find('H1')>=0: 
    if diag_name == '': diag_name = "H1DTacqAxial"
    if shot_number is None: shot_number = 69270


device = pf.getDevice(dev_name)

try:
    old_shot
except:
    old_shot=0


if old_shot>0: # we can expect the variables to be still around, run with -i
    if (old_diag != diag_name) or (old_shot != shot_number): old_shot=0

if old_shot == 0: 
    d = device.acq.getdata(shot_number, diag_name) # ~ 50MB for 6ch 1MS. (27233MP)
    old_shot = shot_number
    old_diag = diag_name

if time_range != None:
    d.reduce_time(time_range)

if start_time is None:
    sv = d.svd()
    sv.svdplot(hold=hold)

else:
    segs = [s for s in d.segment(numpts)]
    segs[i].svd().svdplot()


        
