import pyfusion as pf    # (we will assume these three import lines in all future examples)
import numpy as np       
import matplotlib.pyplot as plt

dev = pf.getDevice('H1Local')  # open the device (choose the experiment)
ne_profile, t_mid, shot = [ ], [ ], [ ]  # prepare empty lists for ne_profile, shot and time of measurement
for shot_number in range(86507, 86517+1):  # the +1 ensures 86517 is the last shot
    d = dev.acq.getdata(shot_number, 'ElectronDensity')	# a multichannel diagnostic
    sections = d.segment(n_samples=1024)   # break in to time segments
    # work through each time segment, extracting the average density during that time
    for seg in sections:
        ne_profile.append(np.average(seg.signal,axis=1))   #  axis=1 -> average over time, not channel
        t_mid.append(np.average(seg.timebase))
    shot.append(shot_number)
# store the data in a DA (Dictionary of Arrays) object, which is like a DataFrame in R or panda
myDA = pf.data.DA_datamining.DA(dict(shot=shot, ne_profile=ne_profile, t_mid=t_mid)) 
myDA.write_arff('ne_profile.arff',['ne_profile'])

