import numpy as np
from pyfusion.data.DA_datamining import Masked_DA, DA
from matplotlib import pyplot as plt



dal = DA ('LP/LP20170920_7_UTDU_2k8.npz')
I0arr = dal['I0']
wbad = np.where((I0arr > .02) | (I0arr < 0))[0]
rejected = float(len(wbad))/len(I0arr.flatten())
print('{n} rejected ({pc:.0f}%)'.format(n=wbad, pc=rejected*100))
I0arr[wbad] = np.nan
IUpper = np.nanmean(I0arr, axis=0)
plt.step(range(len(IUpper)), IUpper)
plt.show()
