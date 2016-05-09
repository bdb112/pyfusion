import numpy as np
import matplotlib.pyplot as plt
from pyfusion.data.DA_datamining import Masked_DA, DA
from pyfusion.data.process_swept_Langmuir import Langmuir_data
import pyfusion

dev_name = "W7X"
diag_name = "W7X_L53_LP1_I"
shot_number = [20160309, 42]

dev = pyfusion.getDevice(dev_name)
echdata = dev.acq.getdata(shot_number,'W7X_TotECH')
probedata = dev.acq.getdata(shot_number, diag_name)
wech = np.where(echdata.signal > 100)[0]
tech = echdata.timebase[wech[0]]
t0_utc = int(tech * 1e9) + echdata.utc[0]
dt = (probedata.utc[0] - t0_utc)/1e9

# gas_g=[3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1]
# time_g=[0, 0.099, 0.1, 0.155, 0.156, 0.162, 0.163, 0.196, 0.197, 0.211, 0.212, 0.24, 0.241, 0.266, 0.267, 0.290, 0.291, 0.327, 0.328, 0.341, 0.342, 0.8]
# Corrected using Maciej's email
time_g = [0, 0.099, 0.1, 0.149, 0.150, 0.160, 0.161, 0.191, 0.191, 0.210, 0.211, 0.230, 0.231, 0.260, 0.261, 0.280, 0.281, 0.320, 0.321, 0.340, 0.341, 0.8]
gas_g = [3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1]
# LP5342.process_swept_Langmuir(t_range=[0.9,1.4],dtseg=.002,initial_TeVpI0=dict(Te=20,Vp=0,I0=None),filename='20160309_42_L532_short')
da532=DA('20160309_42_L532_short.npz')
da57=DA('20160309_42_L572.npz')

plt.plot(da532['t_mid']+dt, da532['ne18'][:,1],hold=0,label='ne18 s3_LP2')
plt.plot(da57['t_mid']+dt, da57['ne'][:,0],label='ne18, s7_LP1')
plt.plot(np.array(time_g), 0.8+(np.array(gas_g)-3),label='gas valve')
plt.plot(echdata.timebase - tech, echdata.signal/1000,label='ECH (MW)',color='magenta', lw=0.3)
plt.legend()
plt.title(da57.name)
plt.xlim(0,0.4)
plt.show()


plt.figure()
da57=DA('20160309_42_L57.npz')
da53=DA('20160309_42_L53.npz')
plt.plot(dt+da53['t_mid'],da53['Te'][:,2], label='Te  s3 LP3')
plt.plot(dt+da53['t_mid'],da53['Te'][:,1], label='Te  s3 LP2')
plt.plot(np.array(time_g), 10*(np.array(gas_g)-3),label='gas valve')
plt.legend()
plt.title(da57.name)
plt.xlim(0,0.4)
plt.show()




"""
gas=np.interp(da532['t_mid'], 0.92+np.array(time_g), gas_g)
da532.plot('ne18',select=[0,1,2,3,4,1],sharex='none')
plt.ylim(0,3)
plt.plot(da532['t_mid'], gas-3)
"""




