""" Example 1, JSPF tutorial: simple density profile scan 

In this file, data was downsampled to save space in the download package.
# The following line was used to downsample the data:
run pyfusion/examples/save_to_local.py shot_list=range(86507,86517+1)  overwrite_local=1 dev_name='H1Local' diag_name='ElectronDensity15' downsample=100 local_dir='pyfusion/examples/JSPF_tutorial/local_data'
"""
import pyfusion as pf    # (we will assume these three import lines in all future examples)
import numpy as np       
import matplotlib.pyplot as plt
import os

plt.figure('Example 1')

dev = pf.getDevice('H1Local')  # open the device (choose the experiment - e.g H-1, LHD, Heliotron-J)
ne_profile, t_mid, shot = [ ], [ ], [ ]  # prepare empty lists for ne_profile, shot and time of measurement

# next line redirects pyfusion to find downsampled local data in ./local_data
pf.config.set('global','localdatapath','local_data') 

for shot_number in range(86507, 86517+1):  # the +1 ensures 86517 is the last shot
        d = dev.acq.getdata(shot_number, 'ElectronDensity15')	# a multichannel diagnostic
        sections = d.segment(n_samples=.001)   # break into time segments
	# work through each time segment, extracting the average density during that time
        for seg in sections:
            ne_profile.append(np.average(seg.signal,axis=1))   #  axis=1 -> average over time, not channel
            t_mid.append(np.average(seg.timebase))
            shot.append(shot_number)

# store the data in a DA (Dictionary of Arrays) object, which is like a DataFrame in R or python panda
myDA = pf.data.DA_datamining.DA(dict(shot=shot, ne_profile=ne_profile, t_mid=t_mid)) 

# the next step - write to arff
myDA.write_arff('ne_profile.arff',['ne_profile'])

# run -i example1a.py                    # to see some plots (note the -i)
