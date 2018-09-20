from __future__ import print_function
from pyfusion.debug_ import debug_
import sys, os
import time as tm

import numpy as np
import matplotlib.pyplot as plt
import pyfusion

from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, Channel, ChannelList, \
    get_coords_for_channel

import MDSplus as MDS
from MDSplus import mdsExceptions as Exc

class W7MDataFetcher(BaseDataFetcher):
    """Fetch the W7X MDS data using thin client"""

    def do_fetch(self):
        sig = self.conn.get(self.mds_path)
        dim = self.conn.get('DIM_OF(' + self.mds_path + ')')
        scl = 1.0
        coords = get_coords_for_channel(**self.__dict__)
        ch = Channel(self.config_name,  coords)
        timedata = dim.data()
        output_data = TimeseriesData(timebase=Timebase(1e-9*timedata),
                                     signal = scl * Signal(sig), channels=ch)
        output_data.meta.update({'shot': self.shot})
        output_data.config_name = self.config_name
        output_data.utc = [timedata[0], timedata[-1]]
        #output_data.units = dat['units'] if 'units' in dat else ''
        debug_(pyfusion.DEBUG, level=1, key='W7M_do_fetch',msg='entering W7X MDS do_fetch')
        return(output_data)
        
    def setup(self):
        """ record some details of the forthcoming fetch so
        that the calling routine in base can give useful error messages
        """
        self.msgs = ''
        self.fetch_mode = 'thin'
        
        debug_(pyfusion.DEBUG, level=1, key='W7M_setup',msg='entering W7X MDS SETUP')
        self.conn = MDS.Connection(self.acq.server)
        print(' path was' , self.conn.get("getenv('qrp_path')"), end=': ')
        self.conn.get('setenv("qrp_path=qrp-server::/w7x/new/qrp;/w7x/vault/qrp")')
        print(' now' , self.conn.get("getenv('qrp_path')"))
        try:
            if 'BRIDGE' in self.config_name:
                catch_exception = Exception
            else:
                catch_exception = ()
            try:
                mdsshot = (self.shot[0] - 20000000) * 1000 + self.shot[1]
                self.msgs += 'try shot ' + str(mdsshot)
                self.conn.openTree(self.tree, mdsshot)
            except catch_exception as reason2:
                print('not found - try for a Lukas test shot')
                mdsshot = (self.shot[0] - 20000000) * 100 + self.shot[1]
                self.msgs += 'try shot ' + str(mdsshot)
                self.conn.openTree(self.tree, mdsshot)

            if hasattr(self, 'roi'):
                ROI = self.roi
            elif hasattr(self.acq, 'roi'):
                ROI = self.acq.roi
            else:
                ROI = None
                
            if ROI is not None:        
                tr_ns = [long(float(t) * 1e9) for t in ROI.split()] 
                context = 'settimecontext({0}Q,{1}Q,{2}Q)'.format(*tr_ns)
                self.conn.get(context)

        except Exc.TreeFILE_NOT_FOUND as reason:
            msg = str("Can't open tree {t} shot {sh} \n {r}"
                      .format(r=str(reason), t=self.device, s=self.shot))
            self.msgs += ': ' + msg
            raise LookupError(msg)
        debug_(pyfusion.DEBUG, level=1, key='W7M_setup',msg='returning from W7X MDS SETUP')
        return(self.conn)

    def error_info(self, step=None):
        """ this puts the device specific info into a string form (mds) to return 
        to the generic caller.
        """
        debug_(pyfusion.DEBUG, level=1, key='error_info',msg='entering error_info')
        msg = 'W7M fetch ' + self.fetch_mode + self.msgs
        
        return(msg)
    
