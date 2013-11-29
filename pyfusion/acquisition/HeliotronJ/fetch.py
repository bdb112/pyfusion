"""Heliotron J data fetchers. """

import tempfile
import numpy as np
import pyfusion
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, ChannelList, Channel
from pyfusion.acquisition.HeliotronJ.make_static_param_db import get_static_params
from pyfusion.debug_ import debug_

try:
    import gethjdata
except:
    import commands, os
    import pyfusion
    print 'Compiling Heliotron J data aquisition library, please wait...'
    cdir = os.path.dirname(os.path.abspath(__file__))
## Note: g77 will do, (remove --fcompiler-g95)  but can't use TRIM function etc 
    for cmd in (
        'f2py --fcompiler=gnu95 -c -m gethjdata -lm -lfdata hj_get_data.f',
        'g77 -Lcdata save_h_j_data.f -lfdata -o save_h_j_data'):
        if pyfusion.VERBOSE > 4: 
            tmp = os.system('cd {cdir}; '.format(cdir=cdir) + cmd)
        else: 
            tmp = commands.getstatusoutput(
                'cd {cdir}; '.format(cdir=cdir) + cmd)
    try:
        print('try after compiling...'),
        import gethjdata
    except:
        raise ImportError, "Can't import Heliotron J data acquisition library"

# Dave had some reason for not including the auto compile - Boyd added 2013
# probably should suppress the auto compile during tests - this was his code.
#except:
#    # don't raise an exception - otherwise tests will fail.
#    # TODO: this should go into logfile
#    print ImportError, "Can't import Heliotron J data aquisition library"

VERBOSE = pyfusion.VERBOSE
OPT = 0

class HeliotronJDataFetcher(BaseDataFetcher):
     """Fetch the HJ data."""

     def do_fetch(self):
         channel_length = int(self.length)
         outdata=np.zeros(1024*2*256+1)
         with tempfile.NamedTemporaryFile(prefix="pyfusion_") as outfile:
             getrets=gethjdata.gethjdata(self.shot,channel_length,self.path,
                                         VERBOSE, OPT,
                                         outfile.name, outdata)
         ch = Channel(self.path,
                      Coords('dummy', (0,0,0)))

         # the intent statement causes the out var to be returned
         # looks like the time,data is interleaved in a 1x256 array
         # it is fed in as real*64, but returns as real*32! (as per fortran decl)
         debug_(pyfusion.DEBUG, 4, key='Heliotron_fetch', msg='after call to getdata')
         output_data = TimeseriesData(timebase=Timebase(getrets[1::2]),
                                 signal=Signal(getrets[2::2]), channels=ch)
         output_data.meta.update({'shot':self.shot})
         if pyfusion.VERBOSE>0: print('HJ config name',self.config_name)
         output_data.config_name = self.config_name         
         output_data.params=get_static_params(shot=self.shot,signal=self.path)[0]
         return output_data

