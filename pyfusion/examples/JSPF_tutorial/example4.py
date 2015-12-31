from pyfusion.data.DA_datamining import DA           
import numpy as np
import matplotlib.pyplot as plt

plt.figure('Example 4')
# Load pre-stored data from 30 shots into a dictionary of arrays (DA)
DA766 = DA('H1_766.npz', load=1)
# Extract all the data into local variables, such as freq, t_mid, phases[0]
DA766.extract(locals())

# Look at one set of instances - probe phase differences - plotting frequency
#   vs time, size indicates amplitude
plt.scatter(t_mid, freq, amp, edgecolor='lightblue')

# Find large amplitude, 'rotating' modes (see text). [0] is required by 
wb = np.where((amp>0.05) & (a12>0.7))[0]     # the numpy where() function

# Overplot with larger symbols proportional to amplitude, coloured by a12
plt.scatter(t_mid[wb], freq[wb], 300*amp[wb], a12[wb])
plt.xlabel(r'$k_H$'); plt.ylabel(r'${\rm frequency/V_{Alfv\'en}}$',fontdict={'fontsize':'large'})
plt.xlim(0,0.042) ; plt.ylim(0, 150)
plt.show()
