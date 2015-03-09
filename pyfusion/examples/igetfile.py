# simple example to illustrate the igetfile class

import pylab as pl
from pyfusion.acquisition.LHD.read_igetfile import igetfile
try: 
    dat=igetfile('/tmp/Flxloop@129000.dat',plot=1)
    figure()
except:
    print('unable to read EG data')
    pl.figure()
try:
    dat=igetfile('/data/datamining/cache/wp/wp@99950.dat.bz2',plot=1)
except:
    print('unable to read file')

# plot the lot
dat.plot()
pl.show()
