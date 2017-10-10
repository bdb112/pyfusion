""" Summarise time validity data in the W7X minerva PARMLOG files
not working yet.
"""
import numpy as np
import matplotlib.pyplot as plt

import pyfusion
from pyfusion.acquisition.W7X.get_url_parms import get_suitable_version

get_suitable_version()
modtimes_list = [[pdict[0],                                               # path
                  int(pdict[1]['parms']['modifiedAt']['dimensions'][0]),  # mod
                  int(pdict[1]['parms']['validSince']['dimensions'][0])]  # valid
                 for pdict in pyfusion.W7X_minerva_cache.items()]
modtimes = np.array(modtimes_list[1:3]).T

by_mod = np.argsort(modtimes[1])

t0 = np.min(modtimes[2])

plt.plot(modtimes[2][by_mod])

plt.show()
