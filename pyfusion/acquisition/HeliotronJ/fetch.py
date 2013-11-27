"""Heliotron J data fetchers. """

import tempfile
import numpy as np
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, ChannelList, Channel
from pyfusion.debug_ import debug_

try:
    import gethjdata
except:
    import commands, os
    import pyfusion
    print 'Compiling Heliotron J data aquisition library, please wait...'
    cdir = os.path.dirname(os.path.abspath(__file__))
## Note: g77 will do, (remove --fcompiler-g95)  but can't use TRIM function etc 
    if pyfusion.VERBOSE > 4: tmp = os.system(
        'cd %s; f2py --fcompiler=gnu95 -c -m gethjdata -lm -lfdata hj_get_data.f' %cdir)
    else: tmp = commands.getstatusoutput(
        'cd %s; f2py --fcompiler=gnu95 -c -m gethjdata -lm -lfdata hj_get_data.f' %cdir)
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

VERBOSE = 0
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

         # looks like the time,data is interleaved float64
         # that seems to be at odds with the apparent real32 declaration in fortran
         # the intent statement causes the out var to be returned
         debug_(pyfusion.DEBUG, 4, 'Heliotron', msg='after call to getdata')
         output_data = TimeseriesData(timebase=Timebase(getrets[1::2]),
                                 signal=Signal(getrets[2::2]), channels=ch)
         output_data.meta.update({'shot':self.shot})
         
         return output_data

