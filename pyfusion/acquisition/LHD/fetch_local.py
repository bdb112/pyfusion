"""LHD data fetchers.
Large chunks of code copied from Boyd, not covered by unit tests,
then copied back to this version to read .npz local data.  THis is different
to reading the files obtained by retrieve
"""
from pyfusion.debug_ import debug_

def newload(filename, verbose=1):
    """ Intended to replace load() in numpy
    """
    from numpy import load as loadz
    from numpy import cumsum
    dic=loadz(filename)
#    if dic['version'] != None:
#    if len((dic.files=='version').nonzero())>0:
    if len(dic.files)>3:
        if verbose>2: print ("local v%d " % (dic['version'])),
    else: 
        if verbose>2: print("local v0: simple "),
        return(dic)  # quick, minimal return

    if verbose>2: print(' contains %s' % dic.files)
    signalexpr=dic['signalexpr']
    timebaseexpr=dic['timebaseexpr']
# savez saves ARRAYS always, so have to turn array back into scalar    
    exec(signalexpr.tolist())
    exec(timebaseexpr.tolist())
    return({"signal":signal, "timebase":timebase, "parent_element": dic['parent_element']})

from os import path
from numpy import mean, array, double, arange, dtype
import numpy as np
import array as Array
import pyfusion as pf

from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, Channel, ChannelList, get_coords_for_channel

VERBOSE = 1
# this form is the default returned by retrieve
#data_filename = "%(diag_name)s-%(shot)d-1-%(channel_number)s"
# this form was used to recover a mistake.
#data_filename = "%(shot)d_%(diag_name)s-%(channel_number)s.npz"
# this form is the way boyd stores them (since 2007...)
data_filename = "%(shot)d_%(config_name)s.npz"  # MP1

class LHDBaseDataFetcher(BaseDataFetcher):
    pass

class LHDTimeseriesDataFetcher(LHDBaseDataFetcher):

#        chnl = int(self.channel_number)
#        dggn = self.diag_name

    def do_fetch(self):
        chan_name = (self.diag_name.split('-'))[-1]  # remove -
        filename_dict = {'shot':self.shot, # goes with Boyd's local stg
                         'config_name':self.config_name}

        #filename_dict = {'diag_name':self.diag_name, # goes with retrieve names
        #                 'channel_number':self.channel_number,
        #                 'shot':self.shot}

        debug_(pf.DEBUG, 4, key='local_fetch')
        for each_path in pf.config.get('global', 'localdatapath').split(':'):
            self.basename = path.join(each_path, data_filename %filename_dict)
    
            files_exist = path.exists(self.basename)
            if files_exist: break

        if not files_exist:
            raise Exception("file {fn} not found. (localdatapath was {p})"
                            .format(fn=self.basename, 
                                    p=pf.config.get('global', 
                                                    'localdatapath').split(':')))
        else:
            signal_dict = newload(self.basename)
            
        if ((chan_name == array(['MP5','HMP13','HMP05'])).any()):  
            flip = -1.
            print('flip')
        else: flip = 1.
        if self.diag_name[0]=='-': flip = -flip
#        coords = get_coords_for_channel(**self.__dict__)
        ch = Channel(self.diag_name,  Coords('dummy', (0,0,0)))
        output_data = TimeseriesData(timebase=Timebase(signal_dict['timebase']),
                                 signal=Signal(flip*signal_dict['signal']), channels=ch)
        # bdb - used "fetcher" instead of "self" in the "direct from LHD data" version
        output_data.config_name = self.config_name  # when using saved files, same as name
        output_data.meta.update({'shot':self.shot})

        return output_data

