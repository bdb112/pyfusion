""" from Example 1, JSPF tutorial: simple density profile scan 

_PYFUSION_TEST_@@shot_range=range(92902,92905+1) dev_name = 'H1Local' diag='ElectronDensity15' dt=0.005
"""
from __future__ import print_function
import pyfusion as pf
import numpy as np       
import matplotlib.pyplot as plt
import os
from pyfusion.data.DA_datamining import DA, Masked_DA
plt.figure('ne_scan')

_var_defaults="""
bads = []
verbose = 0
exclude = ['NE_14', 'NE_5']  # these are masked
dev_name = 'H1'
diag = 'ElectronDensity15'
shot_range=range(92763,92810+1)
shot_range=range(92902,92905+1)
shot_range=range(92919,92966+1)
dt=0.005
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pf.getDevice(dev_name)  # open the device (choose the experiment - e.g H-1, LHD, Heliotron-J)
ne_profile, t_mid, shot = [ ], [ ], [ ]  # prepare empty lists for ne_profile, shot and time of measurement

for shot_number in shot_range:  # the +1 ensures 86517 is the last shot
    try:
        d = dev.acq.getdata(shot_number, diag)	# a multichannel diagnostic
        sections = d.segment(n_samples=dt)   # break into time segments
	# work through each time segment, extracting the average density during that time
        for seg in sections:
            ne_profile.append(np.average(seg.signal,axis=1))   #  axis=1 -> average over time, not channel
            t_mid.append(np.average(seg.timebase))
            shot.append(shot_number)
    except Exception as reason:
        bads.append([shot_number, reason])
        msg = 'skipping shot {s}'
        if verbose>0:
            msg += ' because\n {r}'
        print(msg.format(s=shot_number, r=reason))

# store the data in a DA (Dictionary of Arrays) object, which is like a DataFrame in R or python panda
#myDA = pf.data.DA_datamining.Masked_DA(
#    DAordict = dict(shot=shot, ne_profile=np.array(ne_profile), t_mid=t_mid), valid_keys=['ne_profile'], mask=0)
myDA = DA(dict(shot=shot, ne_profile=np.array(ne_profile), t_mid=t_mid))
myDA.masked = Masked_DA(valid_keys=['ne_profile'], DA=myDA)
myDA.da['mask'] = np.ones(shape=np.shape(myDA[myDA.masked.valid_keys[0]])).astype(np.uint8)
channels = [ch.name.split(':')[-1] for ch in seg.channels]
myDA.infodict.update(dict(channels = channels))
for (c, ch) in enumerate(channels):
    if np.any([ex in ch for ex in exclude]): # if any of exclude in that channel
        myDA.da['mask'][:,c] = 0
myDA.save('/tmp/density_scan')
# the next step - write to arff
# myDA.write_arff('ne_profile.arff',['ne_profile'])

import matplotlib.pyplot as plt
def pl(array, comment=None,**kwargs):
    plt.figure(num=comment)  # coment written to window title
    plt.plot(array, **kwargs)

# get the variables into local scope - so they can be accessed directly
myDA.extract(locals())        

# the ne profiles are in an Nx15 array, where N is the numer of channels
pl(ne_profile[40,:],'one profile')

# plot a sequence of profiles, showing every fifth 
for prof in ne_profile[10:20:5]:
    pl(prof)

# plot all profiles by using the transpose operator to get profiles
pl(ne_profile.T, 'all profiles',color='b',linewidth=.01)

# without the transpose, you will get the time variation for the data
pl(ne_profile, 'time variation, all channels',color='b',linewidth=.3)

# see all profiles as a false colour image
# time and shot number run vertically, each band is a shot
plt.figure(num = 'image of all data')
plt.imshow(ne_profile,origin='lower',aspect='auto')
# the negative data artifacts near (12,200) are due to fringe skips

plt.show(0)    # needed to make figures appear if "run" instead of pasted.

