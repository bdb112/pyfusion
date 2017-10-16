""" Summarise time validity data in the W7X minerva PARMLOG files
    Not sure if there is an implied expiration time?, also puzzled why valid since always matches mod time?
"""
import numpy as np
import matplotlib.pyplot as plt

import pyfusion
from pyfusion.acquisition.W7X.get_url_parms import get_suitable_version, get_parm

# force the cache to be filled
get_suitable_version()
vers = np.array(list(pyfusion.W7X_minerva_cache))
# alternative form:  int(pdict[1]['parms']['validSince']['dimensions'][0])]  # valid
modtimes_list = [[get_parm('parms/modifiedAt/dimensions', pdict)[0],    # [0] -> mod
                  get_parm('parms/validSince/dimensions', pdict)[0],    # [1] -> valid
                  get_parm('parms/validSince/dimensions', pdict)[1]]    # [2] -> valid - to??
                  for pdict in [pyfusion.W7X_minerva_cache[ver] for ver in vers]]
modtimes = np.array(modtimes_list).T

by_mod = np.argsort(modtimes[0])
# see  http://arogozhnikov.github.io/2015/09/29/NumpyTipsAndTricks1.html
ind = np.argsort(by_mod)  # argsort(argsort()) is an amazing way to get the plotting order!
t0 = np.min(modtimes[1])

plt.barh(bottom=ind, left=modtimes[1]-t0,
         tick_label = [ver.split('G_')[1].split('_.')[0] for ver in vers],
         width=modtimes[2]-modtimes[1],  # could clip, but might be misleading
         height=0.8, align='center',color='cyan')
plt.xlim(0, 2*31*24*3600e9) # ~two months
plt.title('validSince tuple in order of modification time')
plt.plot(modtimes[0]-t0, ind, 'or')
for v, ver in enumerate(vers[by_mod]):
    plt.text(0, v, get_parm('parms/generalRemarks/values', pyfusion.W7X_minerva_cache[ver])[0])
    plt.text(0, v - 0.4, get_parm('parms/validSince/values', pyfusion.W7X_minerva_cache[ver])[0])

plt.show()
