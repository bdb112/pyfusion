""" Simple Gaussion Misxture clustering of phase profile data using the supplied data set
See example 6, figure 8. 16 Clusters are allowed for, I chose the two most interesting to plot.
This is not optimised for the number or choice of clusters, but does a reasonable 
job of separating the two main clusters.
It is more effective (e.g. Figure 5) to use von Mises mixtures.  This (vMMM) code
is under development - contact the authors for the latest copy.
 
Takes about 1 minute
Python3 produces a different clustering to Python2 - not sure why?
"""


from sklearn import mixture
from pyfusion.data.DA_datamining import DA, report_mem
import numpy as np
import matplotlib.pyplot as plt
# approx size used in pub  figure(figsize=((11.9,7.6)))
plt.figure('Example 6 - Figure 8')

DA76 = DA('H1_766.npz',load=1)
DA76.extract(locals())

np.random.seed(0)       # ensure the same result (useful for examples)
gmm = mixture.GMM(n_components=16, covariance_type='spherical')
m16 = gmm.fit(phases)   # fit 16 Gaussians
cids = m16.predict(phases)      # assign each point to a cluster id

for c in [7, 9]:    # show the two most interesting clusters in freq vs k_h
                    # select cluster members of sufficient amplitude, a12, and after 5ms
    w = np.where((c == cids) & (amp > 0.08) & (a12 > 0.5) & (t_mid > 0.005))[0]
    # add artificial noise to the k_h value to show points 'hidden' under others
    dither = .008 * np.random.random(len(w))
    # colored by cluster
    plt.scatter(k_h[w] + dither, np.sqrt(ne_1[w])*freq[w], 700*amp[w], 'bgrcmkycmrgrcmykc'[c]) 

plt.ylim(0, 60); plt.xlim(0.2, 1)
plt.xlabel(r'$k_H$', fontdict={'fontsize': 'large'}) ; plt.ylabel('frequency')
plt.show()
