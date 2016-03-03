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
from .get_hj_modules import import_module, get_hj_modules
hjmod, exe_path = get_hj_modules()
import_module(hjmod,'gethjdata',locals())

VERBOSE = pyfusion.VERBOSE

class HeliotronJDataFetcher(BaseDataFetcher):
    """Fetch the HJ data."""

    def do_fetch(self):
        channel_length = int(self.length)
        outdata=np.zeros(1024*2*256+1)
        ##  !! really should put a wrapper around gethjdata to do common stuff          
        #  outfile is only needed if the direct passing of binary won't work
        #  with tempfile.NamedTemporaryFile(prefix="pyfusion_") as outfile:
        ierror, getrets = gethjdata.gethjdata(self.shot,channel_length,self.path,
                                              verbose=VERBOSE, opt=1, ierror=2,
                                              outdata=outdata, outname='')
        if ierror != 0:
            raise LookupError('hj Okada style data not found for {s}:{c}'.format(s=self.shot, c=self.path))

        ch = Channel(self.path,
                     Coords('dummy', (0,0,0)))

        # the intent statement causes the out var to be returned in the result lsit
        # looks like the time,data is interleaved in a 1x256 array
        # it is fed in as real*64, but returns as real*32! (as per fortran decl)
        debug_(pyfusion.DEBUG, 4, key='Heliotron_fetch', msg='after call to getdata')
        # timebase in secs (ms in raw data) - could add a preferred unit?
        # this is partly allowed for in savez_compressed, newload, and
        # for plotting, in the config file.
        # important that the 1e-3 be inside the Timebase()
        output_data = TimeseriesData(timebase=Timebase(1e-3 * getrets[1::2]),
                                 signal=Signal(getrets[2::2]), channels=ch)
        output_data.meta.update({'shot':self.shot})
        if pyfusion.VERBOSE>0: print('HJ config name',self.config_name)
        output_data.config_name = self.config_name         
        stprms = get_static_params(shot=self.shot,signal=self.path)
        if len(list(stprms)) == 0:  # maybe this should be ignored - how do we use it?
            raise LookupError(' failure to get params for {shot}:{path}'
                              .format(shot=self.shot, path=self.path))
        output_data.params = stprms
        return output_data

"""
### Debugging 
Excellent test method for fetch - see below to recompile
1/ (simplest) run manual_recompile.py to compile and check this in one go

2/ manual 
from pyfusion.acquisition.HeliotronJ import fetch
ff=fetch.HeliotronJDataFetcher("HeliotronJ_MP1",58000)
ff.path="MP1"
ff.length=100
ff.do_fetch()

3/ closer to the exes:
from pyfusion.acquisition.HeliotronJ import gethjdata2_7
x=arange(1e6)
gethjdata2_7.gethjdata(58000,100,'DIA135',verbose=1,opt=1,ierror=1,outname='foo',outdata=x)
array([  0.00000000e+00,   1.50000000e+02,  -1.52587891e-04, ...,

and get_static_params
from pyfusion.acquisition.HeliotronJ import make_static_param_db, find_helio_exe
make_static_param_db.get_static_params(58000)
[{'AMPGAIN': 1.0,
  'BITZERO': 32768.0,
  'DATASCL': '0.2                 ',
  'DATAUNIT': 'mWb/V               ',
  'IADC_BIT': 16,
  'IAMPFILE': -11,
etc

To recompile, rename the old .so and then
from pyfusion.acquisition.HeliotronJ import get_hj_modules
get_hj_modules.get_hj_modules
  - or -
run pyfusion/acquisition/HeliotronJ/manual_recompile.py
# which you can edit to put your own test case in
"""
