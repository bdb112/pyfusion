from pyfusion.data.DA_datamining import DA
from matplotlib import pyplot as plt
da9=DA('LP20160309_42_L57_errest_cov_lpf=9.npz')
da99=DA('LP20160309_42_L57_errest_cov_lpf=99.npz')
axs=da9.plot('Te',sharey='all')
for i in range(len(da99['info']['channels'])): axs[i].plot(da99['t_mid'],da99.masked['Te'][:,i])
plt.ylim(0,60)
plt.show(0)
