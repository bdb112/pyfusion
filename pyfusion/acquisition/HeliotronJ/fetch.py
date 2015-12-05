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
# binaries are different for python2 and 3 etc.
# append python version number - but hide '.' 
hj_module  = 'gethjdata'+sys.version[0:3].replace('.','_')
# Note: 3.4 may need f2py3.4 - 3.5 f2py3 gives PyCObject_Type error
f2py = 'f2py3' if sys.version >= '3,0' else 'f2py'

try: 
    # this is the right way, but I can't load mod as name this way
    from importlib import import_module_cant_make_it_wrk
except:
    print("Can't load via official import_module, trying a fudge")
    def import_module(modstr, alt_name=None):
        if alt_name is None:
            exec('import '+modstr)
        else:
            exec('import {m} as {a}'.format(m=modstr, a=alt_name))
"""
#works
exec('import pyfusion.acquisition.HeliotronJ.gethjdata2_7 as gethjdata')
import_module('pyfusion.acquisition.HeliotronJ.gethjdata2_7')
import_module('.gethjdata2_7','pyfusion.acquisition.HeliotronJ')

#doesn't
import_module('gethjdata2_7','pyfusion.acquisition.HeliotronJ.gethjdata2_7')
"""

try:
    import_module(hj_module)
except Exception as reason:
    print("Can't import get_hjdata at first attempt {r}, {args}"
          .format(r=reason, args=reason.args))
# Should use subprocess instead of command, and get more feedback
    import commands, os
    import pyfusion
    print('Compiling Heliotron J data aquisition library, please wait...')
    cdir = os.path.dirname(os.path.abspath(__file__))
## Note: g77 will do, (remove --fcompiler-g95)  but can't use TRIM function etc 
    for cmd in (
        '{f} --fcompiler=gfortran -c -m {m} intel.o libfdata.o -lm  hj_get_data.f'
            .format(m=hj_module, f=f2py),
        'f77 -Lcdata save_h_j_data.f intel.o libfdata.o -o save_h_j_data'  # 2015
    ):
        if pyfusion.VERBOSE > 4: 
            tmp = os.system('cd {cdir}; '.format(cdir=cdir) + cmd)
        else: 
            tmp = commands.getstatusoutput(
                'cd {cdir}; '.format(cdir=cdir) + cmd)
    try:
        print('try after compiling...'),
        import_module(hj_module)
    except Exception as reason:
        print("Can't import get_hjdata at first attempt {r}, {args}"
              .format(r=reason, args=reason.args))
        raise ImportError("Can't import Heliotron J data acquisition library")

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
