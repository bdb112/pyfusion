from sklearn import mixture
from pyfusion.data.DA_datamining import DA, report_mem
import numpy as np
import matplotlib.pyplot as plt
plt.figure('Example 6')

DA76 = DA('H1_766.npz',load=1)
DA76.extract(locals())

np.random.seed(0)       # ensure the same result (useful for examples)
gmm = mixture.GMM(n_components=16, covariance_type='spherical')
m16 = gmm.fit(phases)   # fit 16 Gaussians
cids = m16.predict(phases)      # assign each point to a cluster id

for c in [7, 9]:    # show the two most interesting clusters in freq vs k_h
                    # select cluster members of sufficient amplitude, a12, and after 5ms
    w = np.where((c == cids) & (amp > 0.08) & (a12 > 0.5) & (t_mid > 0.005))[0]
    plt.scatter(k_h[w], np.sqrt(ne_1[w])*freq[w], 300*amp[w], 'bgrcmkycmrgrcmykc'[c]) # colored by cluster

plt.ylim(0, 60); plt.xlim(0.2, 1)
plt.xlabel(r'$k_H$', fontdict={'fontsize': 'large'}) ; plt.ylabel('frequency')
plt.show()
