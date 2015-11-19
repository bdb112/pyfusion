""" Extra example to go with Example 1&2, JSPS tutorial: more realistic density profile scan 
Takes about a minute to run in full on a 2015 model machine
"""
import pyfusion as pf   # (we will assume these three import lines in all future examples)
import numpy as np
import matplotlib.pyplot as plt
plt.figure('Example 2')

dev = pf.getDevice('H1Local')  # open the device (choose the experiment: e.g H-1, LHD, Heliotron-J)
# prepare empty lists for ne_profile, shot and time of measurement
ne_profile, t_mid, shot = [], [], []
# for shot_number in range(86507, 86517+1):  # the +1 ensures 86517 is the last shot
lst = range(83130, 83212+1, 1)
# 133,162,163 weird, 166 172 missing, 171 and 205 channel8 starts late.
for sh in [83133, 83162, 83163, 83166, 83171, 83172, 83205]: lst.remove(sh)
for shot_number in lst:  # the +1 ensures 86517 is the last shot
    d = dev.acq.getdata(shot_number, 'ElectronDensity15')     # a multichannel diagnostic
    sections = d.segment(n_samples=1024)   # break into time segments
    # work through each time segment, extracting the average density during that time
    for seg in sections:
        ne_profile.append(np.average(seg.signal, axis=1))   # axis=1 -> avg over time, not channel
        t_mid.append(np.average(seg.timebase))
        shot.append(shot_number)
# store the data in a DA (Dictionary of Arrays) object, which is like a DataFrame in R or panda
myDA = pf.data.DA_datamining.DA(dict(shot=shot, ne_profile=ne_profile, t_mid=t_mid))
myDA.save('ne_profile2')

###########  Now we will process the data a little before clustering. ###############

# flag all profiles that can't be fitted well by a poly of degree 5
# This will eliminate rubbish data - we record an error, which we can later set a threshold on
deg = 5
x = np.arange(len(ne_profile[0]))
myDA.extract(locals(), limit=6000)  # set limit=2000 to speed up, None to take all data

err = 0 * t_mid                     # prepare an empyy array of the right size to hold errors
for (i, nep) in enumerate(ne_profile):
    p = np.polyfit(x, nep, deg, w=nep)
    err[i] = 1/np.sqrt(len(nep)) * np.linalg.norm(nep*(nep - np.polyval(p, x)))
    small = 0.5  # 0.5 for LHD, 0.2 for H-1
    # disqualify if there are fewer than deg (poly degree) non-small samples
    if len(np.where(nep > small)[0]) < deg:
        err[i] = 9

    # disqualify if the average density is too small
    if np.average(nep) < small/3:
        err[i] = 10

# normalise to the average value, so we can compare shapes
avg = 0 * t_mid   # prepare another variable of the right length
for (i, nep) in enumerate(ne_profile):
    avg[i] = np.average(nep)
    ne_profile[i] = nep/avg[i]

##########  Now we can cluster ############

from scipy.cluster.vq import vq, kmeans, whiten
lw = 0.02  # linewidth - for overplotting many lines, so darkness indicates density
# consider on profiles with low polyfit error (i.e. not really jagged
features = ne_profile[np.where(err<0.2)[0]]
whitened = whiten(features)
# set up the random number generator in a reproducible way - good for examples
# but in reallike, we usually want random starts
from numpy import random
random.seed((1000,2000))

# compute the cluster centres, for 4 clusters (in normalised space)
ccs, distortion = kmeans(whitened, k_or_guess=4)
# get the cluster ids, use the normalised features (whitened)
cids, dist  = vq(whitened, ccs)

cols = ('c,g,r,m,b,y,k,orange,purple,lightgreen,gray'.split(','))  # colours to be rotated thru

# plot all data and overlay the cluster centres, un real units and notmalised
fig,[axunnorm, axnorm] = plt.subplots(1,2)
#axunnorm = None; fig,[[axnorm]] = plt.subplots(1,1,squeeze=False) # to ignore unnormalised

# this order will reproduce the example image, if all the data is used, and seed is set
# but now it doesn't reproduce it exactly.
corder = [1, 3, 2, 0]
plt.rc('font',**{'size':18, 'family':'serif'})

for c in corder:
    ws = np.where(cids == c)[0]
    if axunnorm is not None:   # restore to real space by mult. by avg
        axunnorm.plot(avg[ws]*features[ws].T,cols[c],linewidth=lw)

    axnorm.plot(features[ws].T,cols[c],linewidth=lw)
# and the cluster centres
for c in corder:
    axnorm.plot(ccs[c]*np.std(features,axis=0),'k',linewidth=9,)
    axnorm.plot(ccs[c]*np.std(features,axis=0),cols[c],linewidth=6,)


plt.ylabel(r'$n_e/10^{18}$',fontdict={'fontsize':'large'}) ; plt.xlabel('channel')
plt.xlim(7,15); plt.ylim(0,2)

plt.show()
