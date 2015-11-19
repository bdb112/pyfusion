""" LHD version of example2.py
Takes several minutes.  Need to be on site and have access to LHD data.
Should be able to paste in the next two parts from example2.py after extracting this.
"""
import pyfusion as pf    # (we will assume these three import lines in all future examples)
import numpy as np       
import matplotlib.pyplot as plt
plt.figure('Example 1 - LHD')

from pyfusion.utils import get_local_shot_numbers
shots = np.sort(get_local_shot_numbers('fircall', local_path='/data/datamining/cache/fircall/',number_posn=[-13,-8]))
dev = pf.getDevice('LHD')  # open the device (choose the experiment)
ne_profile, t_mid, shot = [ ], [ ], [ ]  # prepare empty lists for ne_profile, shot and time of measurement
for shot_number in shots:  # the +1 ensures 86517 is the last shot
    d = dev.acq.getdata(shot_number, 'LHD_n_e_array')	# a multichannel diagnostic
    sections = d.segment(n_samples=128)   # break into time segments
    # work through each time segment, extracting the average density during that time
    for seg in sections:
        ne_profile.append(np.average(seg.signal, axis=1))  # axis=1 -> avg over time, not channel
        t_mid.append(np.average(seg.timebase))
        shot.append(shot_number)

# store the data in a DA (Dictionary of Arrays) object, which is like a DataFrame in R or panda
myDA = pf.data.DA_datamining.DA(dict(shot=shot, ne_profile=ne_profile, t_mid=t_mid))
myDA.save('LHD_ne_profile.npz')
myDA.write_arff('ne_profile.arff', ['ne_profile'])

myDA.extract(locals())
