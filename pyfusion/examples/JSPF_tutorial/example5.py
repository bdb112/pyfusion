from pyfusion.data.DA_datamining import DA           
import numpy as np
import matplotlib.pyplot as plt

plt.figure('Example 5')
DA766 = DA('H1_766.npz',load=1)    #  Load data from 30 shots into a dictionary of arrays (DA)
DA766.extract(locals())            #  Extract all the data into local variables, such as freq, t_mid, phases[0]
wb = np.where((amp>0.05) & (a12>0.7))[0]  #  find large amplitude, 'rotating' modes (see text).  

mp = 1.67e-27
mu_0 = 4 * np.pi * 1e-7
V_A = b_0/np.sqrt(mu_0 * n_e * 1e18* mp)  # Calculate VA as a local variable , an array of the same size as the   
                                          #   extracted data
DA766.update(dict(V_A=V_A))        #  Add it to the dictionary - no problem if you use the same name
                                   #   for the dictionary key
DA766.save('H1_766_new')           #  Save the data set (under a new name if you prefer).
                                   #  When you load the dataset in the future, the variable V_A will be there.
plt.plot(k_h, freq*1e3/V_A,'.c')   # symbol '.', colour cyan - we see the values are in the range sqrt beta
plt.plot(k_h[wb], freq[wb]*1e3/V_A[wb],'ob',ms=12) # selecting large amplitude as above, we see beginnings of fig 6
plt.xlabel(r'$k_H$'); plt.ylabel(r'${\rm frequency/V_{Alfven}}$',fontdict={'fontsize':'large'})
plt.show()
# Note - there will be a warning "invalid value encountered in sqrt" due to noise in n_e
# This could be eliminated by selecting using np.where(n_e >= 0)[0]
