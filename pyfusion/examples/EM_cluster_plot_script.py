#_PYFUSION_TEST_@@script
from matplotlib import pyplot as plt
import numpy as np
from numpy import where, array
from pyfusion.data.DA_datamining import DA
from pyfusion.acquisition.read_text_pyfusion import read_text_pyfusion, plot_fs_DA, merge_ds
from pyfusion.visual.window_manager import rmw, cmw, omw, lmw, smw
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.visual.sp import off, on, tog
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden


dd=DA('W7X_MIR/DAMIRNOV_41_13_nocache_15_3m_2018fl.zip')
dd.info()
fig4, axs4=plt.subplots(4,1)
figfs, axf=plt.subplots(1,1)
plot_fs_DA(dd, ax=axf)
w6=where((btw(dd['freq'],4,8)) & (dd['amp']>0.0015) & (dd['t_mid']>3.2) &(dd['shot'] == 180904035))[0]
plot_fs_DA(dd, ax=axf, inds=w6,marker='s',ms=300,alpha=0.5)
axs4[0].plot(array([modtwopi(ph,offset=0) for ph in dd['phases'][w6]]).T,'c',lw=.5)
w6=where((btw(dd['freq'],4,8)) & (dd['amp']>0.015) & (dd['t_mid']>3.2) &(dd['shot'] == 180904035))[0]
axs4[0].plot(array([modtwopi(ph,offset=0) for ph in dd['phases'][w6]]).T,'b',lw=.5)
axs4[0].set_title('4-8kHz, after 3.2s')
w0=where((btw(dd['freq'],0,2)) & (dd['amp']>0.0015) & (dd['t_mid']>3.2) &(dd['shot'] == 180904035))[0]
plot_fs_DA(dd, ax=axf, inds=w0,marker='s',ms=300,alpha=0.5)
axs4[1].plot(array([modtwopi(ph,offset=0) for ph in dd['phases'][w0]]).T,'b',lw=.5)
axs4[1].set_title('0-2kHz, > 3.2s')
w6e=where((btw(dd['freq'],4,12)) & (dd['amp']>0.015) & (dd['t_mid']<3.2) &(dd['shot'] == 180904035))[0]
plot_fs_DA(dd, ax=axf, inds=w6e,marker='^',ms=300,alpha=0.5)
axs4[2].plot(array([modtwopi(ph,offset=0) for ph in dd['phases'][w6e]]).T,'c',lw=.5)
w6e=where((btw(dd['freq'],4,12)) & (dd['amp']>0.025) & (dd['t_mid']<3.2) &(dd['shot'] == 180904035))[0]
axs4[2].plot(array([modtwopi(ph,offset=0) for ph in dd['phases'][w6e]]).T,'b',lw=1)
axs4[2].set_title('4-12kHz, before 3.2s')
axs4[3].plot(array([modtwopi(ph,offset=.5) for ph in dd['phases']])[::5].T,'b',lw=.015)
axs4[3].set_title('all')
plt.show(block=0)
