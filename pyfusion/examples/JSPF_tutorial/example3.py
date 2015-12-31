# example code for figure 3.  The smaller data set in this package produces 
#   different result - see example3_small_data_set.png
# After running example1.py, do
# run -i example3.py
#      Note: if the -i is omitted you will get the message 'np' is not defined
#
# First, flag all profiles that can't be fitted well by a poly
inds = None
deg = 5
x = np.arange(len(ne_profile[0]))
myDA.extract(locals(),inds=inds)  # make sure they are all numpy variables
err = 0 * t_mid
for (i, nep) in enumerate(ne_profile):
    p = np.polyfit(x, nep,deg, w=nep)
    # the error of the polynomial fit
    err[i] = 1/sqrt(len(nep)) * np.linalg.norm(nep*(nep - np.polyval(p, x)))
    small=0.2  # 0.5 for LHD, 0.2 for H-1
    # discard all profiles with too many small data points
    if len(np.where(nep>small)[0]) < deg:
        err[i]=999
    # and discard profiles that are very small
    if np.average(nep)<small/3:
        err[i]=998

# normalise to the average value
avg = 0 * t_mid
for (i, nep) in enumerate(ne_profile):
    avg[i] = np.average(nep)
    ne_profile[i] = nep/avg[i]

# Plot normalised profiles that are reasonably smooth and not discared above
plt.figure(num = "normalised profiles, excluding erroneous data")
for (e,pr) in zip(err,ne_profile):
    if e < small/2:
        plt.plot(pr,color='g',linewidth=.04)
    plt.ylim(0,2)    
plt.show(0)
# the darker areas show recurring profiles.
# We need more information (e.g. power, transform, B_0) to investigate the
# reason for the different profiles.

########## simple clustering ###########
#plt.figure(num='simple vq-means clustering')

from scipy.cluster.vq import vq, kmeans, whiten
features = ne_profile[np.where(err < small/2)[0]]
lw=50./len(features)

whitened = whiten(features)
from numpy import random
random.seed((1000,2000))
codes = 4
code_book, distortion = kmeans(whitened,codes)
code,dist = vq(whitened , code_book)

cols = ('c,g,r,m,b,y,k,orange,purple,lightgreen,gray'.split(',')) # to be rotated

axnorm=None
#fig,[axnorm, axunnorm] = plt.subplots(1,2)
plt.fig,[[axunnorm]] = plt.subplots(1,1,squeeze=False)
corder = range(len(code_book))[::-1]
corder = [1,3,2,0]
for c in corder:
    ws = np.where(code==c)[0]
    if axnorm is not None:
        axnorm.plot(avg[ws]*features[ws].T,cols[c],linewidth=lw)

    axunnorm.plot(features[ws].T,cols[c],linewidth=lw) #,rasterized=True)

for c in corder:
    axunnorm.plot(code_book[c]*std(features,axis=0),'k',linewidth=9,)
    axunnorm.plot(code_book[c]*std(features,axis=0),cols[c],linewidth=6,)
plt.xlim(7,15); plt.ylim(0,2)
plt.show()
