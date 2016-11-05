""" Script to plot probe signals in time for the Marfe shot 52
for Uwe, IAEA
"""
import matplotlib.pyplot as plt


from pyfusion.data.DA_datamining import Masked_DA, DA
from pyfusion.data.signal_processing import smooth_n

plt.rc('font', size=18, weight='normal')
plt.rc('lines', lw=2)


DA523=DA('LP/to_uwe/LP20160309_52_L53_2k2.npz')
DA513=DA('LP/to_uwe/LP20160309_51_L53_2k2.npz')
DA527=DA('LP/to_uwe/LP20160309_52_L57_2k2.npz')
DA517=DA('LP/to_uwe/LP20160309_51_L57_2k2.npz')

DA7 = DA527
fig, axs = plt.subplots(2, 1, sharey='none', sharex='all')
axn, axT = axs
for ch in [1,4,5,6,7]:
  axT.plot(DA7['t'],DA7['Te'][:,ch], label=DA7['info']['channels'][ch],lw=2)    
  axn.plot(DA7['t'],DA7['ne18'][:,ch], label=DA7['info']['channels'][ch],lw=2)    
axn.legend()
axT.legend()
axT.set_ylabel('Te (ev)')
axn.set_ylabel('ne/10^18')
axT.set_ylim(0,40)
axn.set_ylim(0,16)
axn.set_xlim(0.40, 0.65)
fig.subplots_adjust(top=0.95, hspace=0, bottom=0.05)
fig.show()


# Te plots
ch = 6 # 6,3
col = 'b'
for (DA3, DA7) in zip([DA513, DA523], [DA517, DA527]):
    #fig = plt.figure(); ax=plt.gca()
    sht = ''
    # to get combined slide, change to       if col in ['b']
    if col in ['b','g']:
        fig, axs = plt.subplots(2, 1, sharey='all', sharex='all', figsize=[12,6])
    else:
        sht = ', shot ' + str(DA3['info']['shotdata']['shot'][0][1])

    ax3, ax7 = axs
    ax3.set_title(DA3['info']['shotdata']['shot'][0])
    ax3.plot(DA3['t'],DA3['Te'][:,ch], col, lw=0.5)
    ts, ys = smooth_n((DA3['t'],DA3['Te'][:,ch])) 
    ax3.plot(ts, ys, col, label=DA3['info']['channels'][ch] + sht,lw=2)
    ax7.plot(DA7['t'],DA7['Te'][:,ch], col, lw=0.5)
    ts, ys = smooth_n((DA7['t'],DA7['Te'][:,ch]))
    ax7.plot(ts, ys, col, label=DA7['info']['channels'][ch] + sht,lw=2)
    for ax in [ax3, ax7]: 
        leg = ax.legend(prop=dict(size='small'))
        leg.draggable()
        ax.set_ylabel('Te (ev)')
    ax3.set_ylim(0,90)
    ax3.set_xlim(0,0.6)
    ax7.set_xlabel('time (s)')
    fig.subplots_adjust(top=0.9, hspace=0, bottom=0.15)
    fig.show()
    col = 'g'
