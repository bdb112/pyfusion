"""TJ-II data fetchers. """

import numpy as np
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, ChannelList, Channel

try:
    import tjiidata
except:
    # don't raise an exception - otherwise tests will fail.
    # TODO: this should go into logfile
    print  "Can't import TJ-II data aquisition library"
# to use tjii local_data, create a zero length file tjiidata.py (don't add to SVN!)
MAX_SIGNAL_LENGTH = 2**20

class TJIIDataFetcher(BaseDataFetcher):
     """Fetch the TJ-II data."""

     def do_fetch(self):
         print self.shot, self.senal
         data_dim = tjiidata.dimens(self.shot,self.senal)
         if data_dim[0] < MAX_SIGNAL_LENGTH:
             data_dict = tjiidata.lectur(self.shot, self.senal, data_dim[0],data_dim[0],data_dim[1])
         else:
             raise ValueError('Not loading data to avoid segmentation fault in tjiidata.lectur')
         ch = Channel(self.senal,
                      Coords('dummy', (0,0,0)))
         
         if self.invert == 'true': #yuk - # TODO: use boolean type from config
             s = Signal(-np.array(data_dict['y']))
         else:
             s = Signal(np.array(data_dict['y']))
         
         output_data = TimeseriesData(timebase=Timebase(data_dict['x']),
                                      signal=s, channels=ch)
         output_data.meta.update({'shot':self.shot})
         return output_data
