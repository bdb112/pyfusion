"""Heliotron J data fetchers. """

import tempfile
import numpy as np
import pyfusion
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, ChannelList, Channel
from pyfusion.acquisition.HeliotronJ.make_static_param_db import get_static_params
from pyfusion.debug_ import debug_
import sys
# autocompile pulled out to get_hj_modules
from .get_hj_modules import import_module, get_hj_module
hjmod, exe_path = get_hj_module()
import_module(hjmod,'gethjdata')

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

"""
### Debugging 
Excellent test method for fetch
from pyfusion.acquisition.HeliotronJ import gethjdata2_7
x=arange(1e6)
gethjdata2_7.gethjdata(58000,100,'DIA135',verbose=1,opt=0,outname='foo',outdata=x)
array([  0.00000000e+00,   1.50000000e+02,  -1.52587891e-04, ...,

and get_static_params
from pyfusion.acquisition.HeliotronJ import make_static_param_db
make_static_param_db.get_static_params(58000)
[{'AMPGAIN': 1.0,
  'BITZERO': 32768.0,
  'DATASCL': '0.2                 ',
  'DATAUNIT': 'mWb/V               ',
  'IADC_BIT': 16,
  'IAMPFILE': -11,
etc

"""
