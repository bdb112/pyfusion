import numpy as np
import matplotlib.pyplot as plt

import pyfusion as pf
from pyfusion.data.DA_datamining import DA

device_name = 'H1Local'
ne_set = 'ElectronDensity'
shot_range = range(86507, 86517+1)  # the +1 ensures 86517 is the last shot
time_range = [0.02, 0.04]
n_samples = 1024 # define the time interval for averaging n_e 
overlap=1.0      # take an extra n_samples for overlap - 1/2 before and 1/2 after
exception = None

import pyfusion.utils
exec(pf.utils.process_cmd_line_args())

# prepare an empty lists for data - lists are easy to append to
ne_profile = []
shot = [] 
t_mid = []

dev = pf.getDevice(device_name) # open the device (choose the experiment)

for shot_number in shot_range:
    try:
        d = dev.acq.getdata(shot_number, ne_set)
        if time_range != None:
            d.reduce_time(time_range)
        sections = d.segment(n_samples, overlap)
        print(d.history, len(sections))

        for ss,t_seg in enumerate(sections):
            ne_profile.append(np.average(t_seg.signal,axis=1))
            t_mid.append(np.average(t_seg.timebase))

            shot.append(shot_number)
    except exception, reason:
        print 'Error {e} on shot {s}'.format(e=reason, s=shot)
# make the dictionary of arrays and put it in a DA object
myDA = DA(dict(shot=shot, ne_profile=ne_profile, t_mid=t_mid))
myDA.save('ne_profile')



