"""LHD data fetchers.
This gets the data directly from the data server, and so only runs on LHD dmana
v0: Large chunks of code copied from Boyd, not covered by unit tests.
v1: 

See the docstring under class LHDTImeseries
"""
import subprocess
import sys
import tempfile
from os import path, makedirs
import array as Array
from numpy import mean, array, double, arange, dtype, load
import numpy as np
from time import sleep

import pyfusion
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, Channel, ChannelList
from pyfusion.acquisition.LHD.get_basic_diagnostics import get_basic_diagnostics

from pyfusion.debug_ import debug_
#from pyfusion import VERBOSE, DEBUG  really want to import just pyfusion.DEBUG,VERBOSE

this_dir = path.dirname(path.abspath(__file__)) 

data_fileformat = "%(diag_name)s-%(shot)d-1-%(channel_number)s"

class LHDBaseDataFetcher(BaseDataFetcher):

    def error_info(self, step=None):
        """ can only access items that are part of self - others may be volatile
        """
        debug_(pyfusion.DEBUG, level=3, key='error_info',msg='entering error_info')
        """"
        try:
             tree = self.tree
        except:
             try: 
                  tree = self.mds_path_components['tree']
             except:
               tree = "<can't determine>"
               debug_(DEBUG, level=1, key='error_info_cant_determine')

        """
        msg = str("LHDbasedata: Could not open %s, shot %d, channel = %s, step=%s"      
                  %(self.diag_name, self.shot, self.channel_number, step))
        if step == 'do_fetch':
            pass

        #msg += str(" using mode [%s]" % self.fetch_mode)

        return(msg)



class LHDIgetfileReader(LHDBaseDataFetcher):
    """ This uses the igetfile function to return one diagnostic at a time,
    on its own timebase.  The original use of get_basic_diagnostics was to
    get a bunch of diags, on a given timebase.  Might be better to separate the 
    functions in the future.
    Will probably drop the dictionary approach and make each item a separte enetity
    in the .cfg file soon.
    """
    def do_fetch(self):
        if pyfusion.DBG() > 2: print('in fetch',self.diag_name, self)
        debug_(pyfusion.DEBUG, level=3, key='igetfile_fetch')
        diag = self.config_name
        infodict = eval(eval(self.info))
        vardict = get_basic_diagnostics(diags=[diag], times=None, shot=self.shot,
                                        file_info={diag:infodict}, debug=1, 
                                        exception=None)
        debug_(pyfusion.DEBUG, level=2, key='after get_basic')
          
        output_data = TimeseriesData(timebase=Timebase(vardict['check_tm']),
                                     signal=Signal(vardict[self.config_name]),
                                     channels=Channel(self.config_name,Coords('dummy',(0,0,0))))
        output_data.config_name = self.config_name  # ??? bdb - my fault?
        return output_data
     
class LHDTimeseriesDataFetcher(LHDBaseDataFetcher):
     """
     need: export Retrieve=~/retrieve/bin/ # (maybe not) export
     INDEXSERVERNAME=DasIndex.LHD.nifs.ac.jp/LHD

     **Debugging**

     **Off-site**
     in pyfusion::

      # set the config to use LHD fetcher
      pyfusion.config.set('DEFAULT','LHDfetcher','pyfusion.acquisition.LHD.fetch.LHDTimeseriesDataFetcher')
      # choose a shot that doesn't exist locally
      run pyfusion/examples/plot_signals.py shot_number=999 diag_name='VSL_6' dev_name='LHD'

     **On-site**
     test lines for exes::

      retrieve SX8O 74181 1 33
      retrieve Magnetics3lab1 74181 1 33
      2015: retrieve_t seems to only work on FMD
      retrieve_t FMD 117242 1 33 
      different error messages on Magnetics3lab1

     Using retrieve_t::

      Don't know when it is needed -  always trying it first?
      if it gives an error, calculate according to .prm
      timeit fmd=retriever.retrieve('Magnetics3lab1',105396,1,[33],False)
      142ms without retrieve_t, 224 with, including failure (set True in above)

     """
     def do_fetch(self):
          # Allow for movement of Mirnov signals from A14 to PXI crate
        if pyfusion.VERBOSE>1: print('LHDfetch - timeseries')
        chnl = int(self.channel_number)
        dggn = self.diag_name
        if not hasattr(self,'filepath'):
             self.filepath = pyfusion.config.get('global','LHDtmpdata')

        # the clever "-" thing should only be used in members of multi signal diagnostics.
        # so I moved it to base.py.  This means it won't cause sign errors 
        # by doubling up when retrieving from local storage.
        # dggn = (self.diag_name.split('-'))[-1]  # remove -
        debug_(pyfusion.DEBUG, level=5, key='LHD fetch debug') 

        if (dggn == 'FMD'):
            if (self.shot < 72380):
                dggn = 'SX8O'
                if chnl != 0: 
                    chnl = chnl + 33
                    if self.shot < 26208: chnl = chnl +1

        filename_dict = {'diag_name':dggn, 
                         'channel_number':chnl, 
                         'shot':self.shot}
        self.basename = path.join(self.filepath, data_fileformat %filename_dict)

        files_exist = path.exists(self.basename + ".dat") and path.exists(self.basename + ".prm")
        if not files_exist:
            if pyfusion.VERBOSE>3: print('RETR: retrieving %d chn %d to %s' % 
                              (self.shot, int(chnl),
                               self.filepath))
        res = retrieve_to_file(diagg_name=dggn, shot=self.shot, subshot=1, 
                               channel=int(chnl), outdir = self.filepath)
        if not (path.exists(self.basename + ".dat") and 
                path.exists(self.basename + ".prm")):
             raise Exception("something is buggered.")
        self.timeOK = res[3]
        return read_data_from_file(self)



zfile = load(path.join(this_dir,'a14_clock_div.npz'))

a14_clock_div = zfile['a14_clock_div']

def LHD_A14_clk(shot):
    """ Helper routine to fix up the undocumented clock speed changes in the A14"""

    """
    # The file a14_clock_div.npz replaces all this hard coded stuff
    # not sure about the exact turn over at 30240 and many others, not checked above 52k yet
    rate  = array([500,    1000,   500, 1000,    500,   250,  500,     250,   500,   250,   500,   250,   500,   250,   500])
    shots = array([26220, 30240, 30754, 31094, 31315, 49960,  51004, 51330, 51475, 51785, 52010, 52025, 52680, 52690, 52810, 999999])
    where_ge = (shot >= shots).nonzero()[0]
    if len(where_ge) < 1: 
        raise LookupError('a14_clock lookup: shot out of range')

    last_index = max(where_ge)
    rateHz = 1000.*rate[last_index]
    """
    div = a14_clock_div[shot]
    if div > 0: clk = 1e6/div
    else: clk = 0
    rateHz=clk
    # print(rateHz)
    return(rateHz)

def read_data_from_file(fetcher):
    prm_dict = read_prm_file(fetcher.basename+".prm")
    bytes = int(prm_dict['DataLength(byte)'][0])
    bits = int(prm_dict['Resolution(bit)'][0])
    if 'ImageType' not in prm_dict:      #if so assume unsigned
        bytes_per_sample = 2
        dat_arr = Array.array('H')
        offset = 2**(bits-1)
        dtyp = np.dtype('uint16')
    else:
        if prm_dict['ImageType'][0] == 'INT16':
            bytes_per_sample = 2
            if prm_dict['BinaryCoding'][0] == 'offset_binary':
                dat_arr = Array.array('H')
                offset = 2**(bits-1)
                dtyp = np.dtype('uint16')
            elif prm_dict['BinaryCoding'][0] == "shifted_2's_complementary":
                dat_arr = Array.array('h')
                offset = 0
                dtyp = np.dtype('int16')
            # this was added for the VSL digitisers
            elif prm_dict['BinaryCoding'][0] == "2's_complementary": # not sure about this
                dat_arr = Array.array('h')
                offset = 0
                dtyp = np.dtype('int16')
            else: raise NotImplementedError(' binary coding {pd}'.format(pd=prm_dict['BinaryCoding']))

    """
    fp = open(fetcher.basename + '.dat', 'rb')
    dat_arr.fromfile(fp, bytes/bytes_per_sample)
    fp.close()
    """
    dat_arr = np.fromfile(fetcher.basename + '.dat',dtyp)
    #print(dat_arr[0:10])
    #print(fetcher.config_name)


    if fetcher.timeOK:  # we have retrieve_t data!
         #  check for ArrayDataType: float is float32  
         # skip is 0 as there is no initial digitiser type token
         tprm_dict = read_prm_file(fetcher.basename+".tprm",skip=0)
         if pyfusion.VERBOSE>1: print(tprm_dict)
         ftype = tprm_dict['ArrayDataType'][0]
         floats = dict(float = 'float32', double='float64')
         timebase = np.fromfile(fetcher.basename + '.time', 
                                np.dtype(floats[ftype]))

    else:  #  use the info from the .prm file
         clockHz = None

         if 'SamplingClock' in prm_dict: 
             clockHz =  double(prm_dict['SamplingClock'][0])
         if 'SamplingInterval' in prm_dict: 
             clockHz =  clockHz/double(prm_dict['SamplingInterval'][0])
         if 'ClockInterval(uSec)' in prm_dict:  # VSL dig
              clockHz =  1e6/double(prm_dict['ClockInterval(uSec)'][0])
         if 'ClockSpeed' in prm_dict: 
             if clockHz != None:
                 pyfusion.utils.warn('Apparent duplication of clock speed information')
             clockHz =  double(prm_dict['ClockSpeed'][0])
             clockHz = LHD_A14_clk(fetcher.shot)  # see above

         if clockHz != None:
              if 'PreSamples/Ch' in prm_dict:   # needed for "WE" e.g. VSL  
                   pretrig = float(prm_dict['PreSamples/Ch'][0])/clockHz
              else:
                   pretrig = 0.
              timebase = arange(len(dat_arr))/clockHz  - pretrig

         else:  
              debug_(pyfusion.DEBUG, level=4, key='LHD read debug') 
              raise NotImplementedError("timebase not recognised")
    
    debug_(pyfusion.DEBUG, level=4, key='LHD read debug') 
    ch = Channel("{dn}-{dc}" 
                 .format(dn=fetcher.diag_name, 
                         dc=fetcher.channel_number), 
                 Coords('dummy', (0,0,0)))
#    if fetcher.gain != None:   # this may have worked once...not now!
#        gain = fetcher.gain
#    else: 
    #  was - crude!! if channel ==  20: arr = -arr   # (MP5 and HMP13 need flipping)
    try:
        gain = float(fetcher.gain)
    except: 
        gain = 1

    # dodgy - should only apply to a diag in a list - don't want to define -MP5 separately - see other comment on "-"
    #if fetcher.diag_name[0]=='-': flip = -1
    #else: 
    flip = 1

    # not sure if this needs a factor of two for RangePolarity,Bipolar (A14)
    rng=None
    for key in 'Range,Range(V)'.split(','):  # equivalent alteratives
         rng=prm_dict.get(key)
         if rng is not None: break

    scale_factor = flip*double(rng[0])/(2**bits)
    # not sure how this worked before I added array() - but has using
    # array slowed things?  I clearly went to trouble using tailored ints above?
    # - yes array(dat_arr) takes 1.5 sec for 4MS!!
    # Answer - using numpy fromfile('file',dtype=numpy.int16) - 16ms instead!
    # NOTE! ctype=int32 required if the array is already an np array - can be fixed once Array code is removed (next version)
    output_data = TimeseriesData(timebase=Timebase(timebase),
                                 signal=Signal(scale_factor*gain*(array(dat_arr,dtype=np.int32)-offset)),
                                 channels=ch)
    #print(output_data.signal[0:5],offset,(array(dat_arr)-offset)[0:5])
    output_data.meta.update({'shot':fetcher.shot})
    output_data.config_name = fetcher.config_name
    output_data.params = prm_dict
    return output_data


def read_prm_file(filename,skip=1):
    """ Read a prm file into a dictionary.  Main entry point is via filename,
    possibly reserve the option to access via shot and subshot
    >>> pd = read_prm_file(filename=filename)
    >>> pd['Resolution(bit)']
    ['12', '4']
    skip=0 for .tprm files, which omit the 'digitiser' token

    This comes from the line
    Aurora14,Resolution(bit),12,4
    where (maybe?) last digit is 1: string, 4: mediumint, 5: float, 6 signed int, 7, bigint??
    """
    f = open(filename)
    prm_dict = {}
    for s in f:
        s = s.strip("\n")
        toks = s.split(',')  
        if len(toks)<2: print('bad line %s in %f' % (s, filename))
        key = toks[skip]
        prm_dict.update({key: toks[skip+1:]})
    f.close()
    return prm_dict

def persevere_with(cmd, attempts=2, quiet=False):
    attempt = 1
    while(1):
         if attempt>1: print('attempt {a}, {c}'.format(a=attempt, c=cmd))
         retr_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
         (resp,err) = retr_pipe.communicate()
         if (err != '') or (retr_pipe.returncode != 0): 
              attempt += 1
              if pyfusion.VERBOSE > 0:
                   print('response={resp}, errmsg={err}, after {attempt}attempts'
                         .format(resp=resp,err=err,attempt=attempt)) #,
              if ((attempt>attempts) or ("not exist" in err) 
                  or ( "data not found" in resp)
                  or ('retrieveGetDTSdatax2' in resp)): 
                   if not quiet:
                        raise LookupError(str("Error {p} accessing retrieve:"
                                              "cmd={c} \nstdout={r}," 
                                              "stderr={e}, attempt {a}"
                                              .format(p =retr_pipe.poll(), 
                                                      c= cmd, r=resp, e=err, 
                                                      a=attempt)))
                   return(False, resp, err)  # False = failure
              sleep(2)
         else:
              return(True, resp, err)



def retrieve_to_file(diagg_name=None, shot=None, subshot=None, 
                     channel=None, outdir = None, get_data=True):
    """ run the retrieve standalone program to get data to files,
    and/or extract the parameter and summary information.

    Retrieve Usage from Oct 2009 tar file:
    Retrieve DiagName ShotNo SubShotNo ChNo [FileName] [-f FrameNo] [-h TransdServer] [-p root] [-n port] [-w|--wait [-t timeout] ] [-R|--real ]
   """
    from pyfusion.acquisition.LHD.LHD_utils import get_free_bytes, purge_old

# The old pyfusion used None to indicate this code could choose the location
# in the new pyfusion, it is fixed in the config file.
#    if outdir is None: 
#        outdir = tempfile.gettempdir() + '/'

    if not(path.exists(outdir)): makedirs(outdir)

    freebytes=get_free_bytes(outdir)
    if freebytes < pyfusion.TMP_FREE_BYTES:
         try_for = 100  # go for 100 files, assum 500k on average
         purge_old(outdir, '*',try_for)  #dat')  # ONLY DO .DAT have to manually purge prm
         if (get_free_bytes(outdir) < freebytes*(try_for * 500e3)):
              pyfusion.logger.warning("Warning - unable to clear much space! {fGb:.1f}Gb free".format(fGb=freebytes/1e9))

#
    cmd = str("retrieve %s %d %d %d %s" % (diagg_name, shot, subshot, channel, path.join(outdir, diagg_name)))

    if (pyfusion.VERBOSE > 1): print('RETR: %s' % (cmd))

    (dataOK, resp, err) = persevere_with(cmd, 10, quiet=False)  # these errors are bad
    cmd_t = cmd.replace('retrieve', 'retrieve_t')      
    (timeOK, trep, terr) = persevere_with(cmd_t, 10, quiet=True)  # error => retrieve_t N/A

    fileroot = ''
    for lin in resp.split('\n'):
         if pyfusion.DBG() > 3: print('*******',lin)
         if lin.find('parameter file')>=0:
              fileroot = lin.split('[')[1].split('.prm')[0]
    if fileroot == '':
         raise LookupError('parameter file not found in <<{r}>>'.format(r=resp))

    return(resp, err, fileroot, timeOK)
