import numpy as np
import matplotlib.pyplot as plt
from pyfusion.data.DA_datamining import Masked_DA, DA
from pyfusion.data.process_swept_Langmuir import Langmuir_data
import pyfusion
from lukas.PickledAssistant import lookupPosition

def correlation(x,y,AC=True,coefft=True):
    """  AC => remove means first
         coefft: True is dimensionless - else return the amplitude
    """
    import numpy as np
    clate = np.correlate
    mean = np.mean
    if AC:
        x = x - mean(x)
        y = y - mean(y)
    if coefft:
        return(clate(x,y)/np.sqrt(clate(x,x) * clate(y,y)))
    else:
        return(clate(x,y)/np.sqrt(clate(x,x) * clate(x,x)))
        


dev_name = "W7X"
diag_name = "W7X_L53_LP01_I"
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
time_g = np.array([0, 0.099, 0.1, 0.149, 0.150, 0.160, 0.161, 0.191, 0.191, 0.210, 0.211, 0.230, 0.231, 0.260, 0.261, 0.280, 0.281, 0.320, 0.321, 0.340, 0.341, 0.8])
gas_g = np.array([3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1, 3.7, 3.7, 3.1, 3.1])
# LP5342.process_swept_Langmuir(t_range=[0.9,1.4],dtseg=.002,initial_TeVpI0=dict(Te=20,Vp=0,I0=None),filename='20160309_42_L532_short')
#da532=DA('20160309_42_L532_short.npz')
da532=DA('LP/LP20160309_42_L53_2k2.npz')
#da532=DA('LP/LP20160309_52_L53_2k2.npz')
#da532=DA('LP/LP20160309_44_L53.npz')
da572=DA('LP/LP20160309_42_L57_2k2.npz')
#da572=DA('LP/LP20160309_52_L57_2k2.npz')


# offsets for medium text, not including dot size, second element is the probe number to avoid
voffs = dict(LP17=[.002,20], LP10=[.002, 6], LP09=[0.001,5], LP06=[-.001,1], LP12=[.002,14])

fig,(axcorr,axmap) = plt.subplots(2,1) 
da = da532
chans = np.array(da['info']['channels'])
from scipy.interpolate import griddata
grid_t = np.mgrid[0:1:3000j]
grid_gas = griddata(time_g, gas_g, (grid_t), method='linear')
w = np.where((grid_t<0.356) & (grid_t>0.18))[0]
corrs = []
corr_diag = 'ne18'
coefft = 1  # 1 for dimensionless, 0 for phys units

# scan over a range of delays
for ch in range(len(da['info']['channels'])):
    corr = []
    # t_range = np.linspace(-.04,0.04,50) # too big - leads to incoherence
    t_range = np.linspace(-.01,0.035,50)
    for toffs in t_range:
        grid_ne = griddata(dt-toffs+da['t_mid'], da[corr_diag][:,ch], (grid_t), method='linear')
        # plt.plot(grid_t,grid_ne,'.')
        corr.append(correlation(grid_gas[w], grid_ne[w],coefft=coefft)[0])
    corrs.append(corr)
for (c, corr) in enumerate(corrs):
    axcorr.plot(t_range,corr,label=chans[c][1:],linestyle=['-','--'][c>10])
leg = axcorr.legend(prop=dict(size='small'))
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
avg_ne= np.mean(grid_ne[w])
avg_gas= np.mean(grid_gas[w])

corr = np.correlate(grid_gas[w]-avg_gas, grid_ne[w]-avg_ne)/np.sqrt(np.correlate(grid_gas[w]-avg_gas, grid_gas[w]-avg_gas)*np.correlate(grid_ne[w]-avg_ne, grid_ne[w]-avg_ne))
avg_corr_coeff = np.array([np.sum(corrs[i]*np.sign(t_range-.012))/(2*len(corrs)/np.pi) for i in range(len(corrs))])
X, Y, Z = np.array(da['info']['coords']).T
th = np.deg2rad(-18)
x = X * np.cos(th) - Y*np.sin(th)
y = X * np.sin(th) + Y*np.cos(th)
z = Z
title = str('{n}: {seg} segment'.format(n=da.name, seg=['upper','lower'][z[0]<0]))
scl = 1000
wnan = np.where(np.isnan(avg_corr_coeff))[0]
avg_corr_coeff[wnan] = 0
sp = axmap.scatter(x,z,scl*np.abs(avg_corr_coeff),avg_corr_coeff,cmap='bwr',vmin=-.7,vmax=.7)
fig.colorbar(sp)
axmap.plot([0, 0], axmap.get_ylim(), linewidth=.3)
dot_radius = 2e-4 * np.sqrt(sp.get_sizes())
for (c, ch) in enumerate(chans):
    voff = 0 
    for k in voffs:
        if k in ch:
            offender = [off for off in chans if format(voffs[k][1],'02d') in off][0]
            o = np.where(offender == chans)[0]
            voff = np.sign(z[c])*(voffs[k][0] + 0.7*dot_radius[o]*np.sign(voffs[k][0]))
            print(offender, o, dot_radius[o])
    plt.text(x[c] + (dot_radius[c] +.001)*np.array([1,-1])[x[c]<0], z[c] + voff, ch[2:],
             fontsize='medium', horizontalalignment=['left','right'][x[c]<0], verticalalignment='center')

plt.xlim(-0.08,0.08)
#plt.ylim(0.16,0.25)
axmap.set_aspect('equal')
ylab = str('correlation with {corr_diag} {typ}'
           .format(corr_diag=corr_diag, typ = [' (phys units)',' coefft'][coefft]))
axcorr.set_ylabel(ylab)
axmap.set_ylabel('Z(m)')
axmap.set_xlabel('horizontal distance from limiter midplane (m)')
fig.suptitle(title)
axcorr.set_xlabel('time delay (s)')
fig.subplots_adjust(left=0.1, right=0.95,top=0.96,bottom=0.07, hspace=.15)
plt.show()

figLC, axLC = plt.subplots(1,1)
distLC = []
for (c,ch) in enumerate(chans):
    LPnum = int(ch[-2:])
    lim = 'lower' if 'L57' in ch else 'upper'
    X,Y,Z,dLC = lookupPosition(LPnum, lim)
    distLC.append(dLC)
axLC.plot(np.sign(x)*distLC, avg_corr_coeff,'o')
axLC.set_xlabel('distance to LCFS (-ve for left side)')
axLC.set_ylabel(ylab)
figLC.show()

"""
"""
plt.figure()
ch1=0
plt.plot(da572['t_mid']+dt, da572['ne18'][:,ch1],hold=0,label='ne18 s'+ da572['info']['channels'][ch1][2:])
ch2=1
plt.plot(da532['t_mid']+dt, da532['ne18'][:,ch2],label='ne18 s'+ da532['info']['channels'][ch2][2:])
plt.plot(np.array(time_g), 0.8+(np.array(gas_g)-3),label='gas valve')
plt.plot(echdata.timebase - tech, echdata.signal/1000,label='ECH (MW)',color='magenta', lw=0.3)
plt.legend()
plt.title(da572.name)
plt.xlim(0,0.4)
plt.xlabel('time after ECH start')
plt.show()


plt.figure()
da57=DA('LP/LP20160309_42_L57_2k2.npz')
da53=DA('LP/LP20160309_42_L53_2k2.npz')
ch3=2
plt.plot(dt+da53['t_mid'],da53.masked['Te'][:,ch3], label='Te s'+da53['info']['channels'][ch3][2:])
ch4=1
plt.plot(dt+da53['t_mid'],da53.masked['Te'][:,ch4], label='Te s'+da53['info']['channels'][ch4][2:])
plt.plot(np.array(time_g), 10*(np.array(gas_g)-3),label='gas valve')
plt.legend()
plt.title(da53.name)
plt.xlim(0,0.4)
plt.xlabel('time after ECH start')
plt.show()
"""
"""
"""
gas=np.interp(da532['t_mid'], 0.92+np.array(time_g), gas_g)
da532.plot('ne18',select=[0,1,2,3,4,1],sharex='none')
plt.ylim(0,3)
plt.plot(da532['t_mid'], gas-3)
"""




