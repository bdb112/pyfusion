# trivial example to illustrate the igetfile class

import pylab as pl
import os
import traceback

from pyfusion.acquisition.LHD.read_igetfile import igetfile
testfile = os.path.join(os.path.dirname(__file__),'wp@54185.dat.bz2')

try: 
    dat=igetfile(testfile,plot=1)
    pl.figure()
    dat.plot()
except Exception as reason:
    print('unable to read EG data from examples - maybe no files there "{r}"'.format(r=reason))
    pl.figure()
try:
    # nice if we could always have it in this place (may be a soft link(
    dat=igetfile(os.getenv('HOME')+'/data/datamining/cache/wp/wp@99950.dat.bz2',plot=1)
except Exception as reason:
    print('unable to read file - maybe no files there "{r}"'.format(r=reason))
    traceback.print_exc()

# plot the lot
pl.figure()
dat.plot()
pl.show(block=0)
