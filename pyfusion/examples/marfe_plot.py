import numpy as np
import matplotlib.pyplot as plt
from pyfusion.data.DA_datamining import Masked_DA, DA
from pyfusion.data.process_swept_Langmuir import Langmuir_data
import pyfusion

_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_L53_LP01_I"
shot_number = [20160218, 5]
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)
echdata = dev.acq.getdata(shot_number,'W7X_TotECH')
ECEdata = dev.acq.getdata(shot_number,'W7X_ECE_13')
probedata = dev.acq.getdata(shot_number, diag_name)
wech = np.where(echdata.signal > 100)[0]
tech = echdata.timebase[wech[0]]
t0_utc = int(tech * 1e9) + echdata.utc[0]
dt = (probedata.utc[0] - t0_utc)/1e9
dtece = (ECEdata.utc[0] - t0_utc)/1e9

# gas_g=[3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1]
# time_g=[0, 0.099, 0.1, 0.155, 0.156, 0.162, 0.163, 0.196, 0.197, 0.211, 0.212, 0.24, 0.241, 0.266, 0.267, 0.290, 0.291, 0.327, 0.328, 0.341, 0.342, 0.8]
# Corrected using Maciej's email
time_g = [0, 0.099, 0.1, 0.149, 0.150, 0.160, 0.161, 0.191, 0.191, 0.210, 0.211, 0.230, 0.231, 0.260, 0.261, 0.280, 0.281, 0.320, 0.321, 0.340, 0.341, 0.8]
gas_g = [3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1]
# LP5342.process_swept_Langmuir(t_range=[0.9,1.4],dtseg=.002,initial_TeVpI0=dict(Te=20,Vp=0,I0=None),filename='20160218_5_L532_short')
#da532=DA('20160218_5_L532_short.npz')
base = str('LP/LP{d}_{s}_L'.format(d=shot_number[0], s=shot_number[1]))
# fall back to 4ms versions if no 2ms version
try:
    da532 = DA(base+'532.npz')
except IOError:
    da532 = DA(base+'53.npz') 
try:
    da572 = DA(base+'572.npz')
except IOError:
    da572 = DA(base+'57.npz')

ch1=0
plt.plot(da572['t_mid']+dt, da572['ne18'][:,ch1],hold=0,label='ne18 s'+ da572['info']['channels'][ch1][3:])
ch2=1
plt.plot(da532['t_mid']+dt, da532['ne18'][:,ch2],label='ne18 s'+ da532['info']['channels'][ch2][3:])
#plt.plot(np.array(time_g), 0.8+(np.array(gas_g)-3),label='gas valve')
plt.plot(ECEdata.timebase + dtece, ECEdata.signal/1000,label='ECE_13',color='red', lw=0.3)
plt.plot(echdata.timebase - tech, echdata.signal/1000,label='ECH (MW)',color='magenta', lw=0.3)
plt.legend()
plt.title(da572.name)
plt.xlim(-0.01,1)
plt.show()


plt.figure()
da57=DA(base + '57.npz')
da53=DA(base + '53.npz')
ch3=2
plt.plot(dt+da53['t_mid'],da53['Te'][:,ch3], label='Te s'+da53['info']['channels'][ch3][3:])
ch4=1
plt.plot(dt+da53['t_mid'],da53['Te'][:,ch4], label='Te s'+da53['info']['channels'][ch4][3:])
#plt.plot(np.array(time_g), 10*(np.array(gas_g)-3),label='gas valve')
plt.legend()
plt.title(da53.name)
plt.xlim(-0.01,1)
plt.show()




"""
gas=np.interp(da532['t_mid'], 0.92+np.array(time_g), gas_g)
da532.plot('ne18',select=[0,1,2,3,4,1],sharex='none')
plt.ylim(0,3)
plt.plot(da532['t_mid'], gas-3)
"""




