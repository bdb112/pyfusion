# Example code to accompany figure 6, the python k means version of Example 2.
#  The smaller data set in this package produces a slighly different result than the 
#  full data set in the article - see example2_small_data_set.png
#  The code for the actual figure is in example2a.py
#  After running example1.py, do
# run -i example2.py
#      Note: if the -i is omitted you will get the message 'np' is not defined
#
import sys
try:  # catch error if user forgets to set up the data first, make it available to this with -i
    x = np.arange(len(ne_profile[0]))  
    myDA.extract(locals())  # makes sure they are all numpy floating point arrays
except NameError as reason:
    print('please run example1 first, THEN run -i example2   (so that myDA is carried over)')
    sys.exit()


###########  Now we will process the data a little before clustering. ###############

# flag all profiles that can't be fitted well by a poly of degree 5
# This will eliminate rubbish data - we record an error, which we can later set a threshold on

deg = 5
err = 0 * t_mid                     # prepare an empty array of the right size to hold errors
for (i, nep) in enumerate(ne_profile):
    p = np.polyfit(x, nep, deg, w=nep)
    err[i] = 1/np.sqrt(len(nep)) * np.linalg.norm(nep*(nep - np.polyval(p, x)))
    small = 0.2  # 0.5 for LHD, 0.2 for H-1
    # disqualify if there are fewer than deg (poly degree) non-small samples
    if len(np.where(nep > small)[0]) < deg:
        err[i] = 9                  #  (flag as bad by setting large error)

    # disqualify if the average density is too small
    if np.average(nep) < small/3:
        err[i] = 10

# normalise to the average value, so we can compare shapes
avg = 0 * t_mid   # prepare another variable of the right length
for (i, nep) in enumerate(ne_profile):
    avg[i] = np.average(nep)
    ne_profile[i] = nep/avg[i]

# Plot only the normalised profiles that are reasonably smooth and not discarded above
plt.figure(num = "normalised profiles, excluding erroneous data")
for (e,pr) in zip(err,ne_profile):
    if e < small/2:
        plt.plot(pr,color='g',linewidth=.04)
    plt.ylim(0,2)    
plt.show(0)
# The darker areas show recurring profiles.
# We would need more information (e.g. power, transform, B_0) to investigate the
#   reason for the different profiles.

########## simple k-means clustering ###########


from scipy.cluster.vq import vq, kmeans, whiten
# consider only profiles with low polyfit error (i.e. not really jagged
features = ne_profile[np.where(err < small/2)[0]]
lw=50./len(features)     # linewidth - for overplotting many lines, so darkness indicates density

whitened = whiten(features)
# set up the random number generator in a reproducible way - good for examples
# but in real life, we usually want random starts
from numpy import random
random.seed((1000,2000))
# compute the cluster centres, for 4 clusters (in normalised space)
ccs, distortion = kmeans(whitened, k_or_guess=4)
# get the cluster ids, use the normalised features (whitened)
cids, dist  = vq(whitened, ccs)

cols = ('c,g,r,m,b,y,k,orange,purple,lightgreen,gray'.split(','))  # colours to be rotated thru

# plot all data and overlay the cluster centres, in real units and normalised
# comment one of the following two lines
fig,[axunnorm, axnorm] = plt.subplots(1,2)     # this shows both norm and unnorm
#axunnorm = None; fig,[[axnorm]] = plt.subplots(1,1,squeeze=False) # this ignore unnormalised

# this order will reproduce the example image, if all the data is used, and seed is set
# but now it doesn't reproduce it exactly.
corder = [1, 3, 2, 0]
plt.rc('font',**{'size':18, 'family':'serif'})

# plot, colouring according to cluster membership
for c in corder:
    ws = np.where(cids == c)[0]
    if axunnorm is not None:   # restore to real space by mult. by avg
        axunnorm.plot(avg[ws]*features[ws].T,cols[c],linewidth=lw)

    axnorm.plot(features[ws].T,cols[c],linewidth=lw)
# and the cluster centres
for c in corder:
    axnorm.plot(ccs[c]*np.std(features,axis=0),'k',linewidth=9,)     #  outline effect
    axnorm.plot(ccs[c]*np.std(features,axis=0),cols[c],linewidth=6,) #  color


plt.ylabel(r'$n_e/10^{18}$',fontdict={'fontsize':'large'}) ; plt.xlabel('channel')
plt.xlim(7,15); plt.ylim(0,2)

plt.show()
