# simple example to illustrate the igetfile class

import pylab as pl
import os

from pyfusion.acquisition.LHD.read_igetfile import igetfile
try: 
    dat=igetfile('/tmp/Flxloop@129000.dat',plot=1)
    pl.figure()
    dat.plot()
except:
    print('unable to read EG data')
    pl.figure()
try:
    # nice if we could always have it in this place (may be a soft link(
    dat=igetfile(os.getenv('HOME')+'/data/datamining/cache/wp/wp@99950.dat.bz2',plot=1)
except:
    print('unable to read file')

# plot the lot
pl.figure()
dat.plot()
pl.show()
