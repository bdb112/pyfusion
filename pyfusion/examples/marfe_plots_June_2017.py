"""from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup
t_range_list_5marfe=[[0.3,0.32], [0.45,0.47], [0.5,0.52], [.53,0.54], [0.5,0.52]]
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list="['LP/all_LD/LP20160224_30_L5SEG_2k2.npz']" labelpoints=0 t_range_list=[0.1,0.2] diag2=ne18 av=np.median
sigs224_30=sigs
sigs3020=np.array([sigs224_30[1]*exp(abs(solds[1])/20),sigs224_30[3]*exp(abs(solds[3])/20)])/6
sigs3015=np.array([sigs224_30[1]*exp(abs(solds[1])/15),sigs224_30[3]*exp(abs(solds[3])/15)])/6
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list="['LP/all_LD/LP20160309_10_L5SEG_2k2.npz']" labelpoints=0 t_range_list=[0.1,0.2] diag2=ne18 av=np.median nrms=sigs3015
axLCne.plot(sort(sold), 3.2*exp(-abs(sort(sold)/15)),'--',label='Feng',lw=2)
axLCTe.plot(sort(sold), 60*exp(-abs(sort(sold)/30))'--',label='Feng',lw=2)
plt.legend()
plt.show()
axLCTe.plot(sort(sold), 60*exp(-abs(sort(sold)/30)),'--',label='Feng',lw=2)
plt.show()
%hgrepmagic python/
"""
"""
_PYFUSION_TEST_@@SCRIPT
Try to reproduce work in these two sessions
==== session 4081, 2016-12-06 10:32:55    =====
==== session 5336, 2020-03-27 18:17:59    =====
"""

from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup
#sys.path.append('/home/bdb112/python/')
t_range_list_5marfe=[[0.3,0.32], [0.45,0.47], [0.5,0.52], [.53,0.54], [0.5,0.52]]
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list="['LP/all_LD/LP20160224_30_L5SEG_2k2.npz']" labelpoints=0 t_range_list=[0.1,0.2] diag2=ne18 av=np.median use_tmid=1
sigs224_30=sigs
sigs3020=np.array([sigs224_30[1]*exp(abs(solds[1])/20),sigs224_30[3]*exp(abs(solds[3])/20)])/6
sigs3015=np.array([sigs224_30[1]*exp(abs(solds[1])/15),sigs224_30[3]*exp(abs(solds[3])/15)])/6
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list="['LP/all_LD/LP20160309_10_L5SEG_2k2.npz']" labelpoints=0 t_range_list=[0.1,0.2] diag2=ne18 av=np.median nrms=sigs3015
axLCTe.plot(sort(sold), 60*exp(-abs(sort(sold)/30)),'--',label='Feng',lw=2)
plt.legend()
plt.show()
