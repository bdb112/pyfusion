#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys, json, time

from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

if len(sys.argv)>1:
    fn = sys.argv[1]
    dat = json.load(open(fn))
    dim = np.clip(np.array(dat['dimensions']) - dat['dimensions'][0],0,1e99) 
    if len(sys.argv)>2:
        stp = int(sys.argv[2])
    else:
        stp = max(1, len(dim)//20000)

    plt.plot(1e-9*dim[::stp], dat['values'][::stp])
    plt.title('{fn}: {tm}: {n:,} samples'.
              format(fn=fn, n=len(dim), tm=time.asctime(time.gmtime(float(dat['dimensions'][0]//1000000000)))))
    plt.show()
