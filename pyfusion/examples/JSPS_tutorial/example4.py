from pyfusion.data.DA_datamining import DA           
import numpy as np
import matplotlib.pyplot as plt

plt.figure('Example 4')
DA766 = DA('H1_766.npz',load=1)    # load data from 30 shots into a dictionary of arrays (DA)
DA766.extract(locals())            #  extract all the data into local variables, such as freq, t_mid, phases[0]
                                   # look at one instance - set of probe phase differences
plt.scatter(t_mid, freq, amp)      # plot frequency vs time, size indicates amplitude
wb = np.where((amp>0.05) & (a12>0.7))[0]  #  find large amplitude, 'rotating' modes (see text).  
                                   # The [0] is required by the numpy where() function
plt.scatter(t_mid[wb], freq[wb], 300*amp[wb], a12[wb])   # overplot these with larger symbols coloured by a12
plt.xlabel(r'$k_H$'); plt.ylabel(r'${\rm frequency/V_{Alfven}}$',fontdict={'fontsize':'large'})
plt.show()
