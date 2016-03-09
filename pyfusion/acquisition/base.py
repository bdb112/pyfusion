"""Base classes for  pyfusion data acquisition. It is  expected that the
base classes  in this module will  not be called  explicitly, but rather
inherited by subclasses in the acquisition sub-packages.

"""
from numpy.testing import assert_array_almost_equal
from numpy import ndarray, shape
from pyfusion.conf.utils import import_setting, kwarg_config_handler, \
     get_config_as_dict, import_from_str
from pyfusion.data.timeseries import Signal, Timebase, TimeseriesData
from pyfusion.data.base import Coords, Channel, ChannelList
from pyfusion.debug_ import debug_
import traceback
import sys, os
import pyfusion  # only needed for .VERBOSE and .DEBUG

def newloadv3(filename, verbose=1):
    """ This the the version that works in python 3, but can't handle Nans
    Intended to replace load() in numpy
    counterpart is data/savez_compress.py
    """
    from numpy import load as loadz
    from numpy import cumsum, array
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
    if 'time_unit_in_seconds' in dic:
        timeunitexpr = dic['time_unit_in_seconds']
    else:
        timeunitexpr = array(1)

    # savez saves ARRAYS always, so have to turn array back into scalar    
    # exec(signalexpr.tolist())
    # Changed exec code to eval for python3, otherwise the name was not defined
    #   for the target variables - they could only be accessed with 
    #   e.g. locals().signal 
    # retdic = {"signal":locals()['signal'], "timebase":locals()['timebase'], 
    #           "parent_element": dic['parent_element']}
    # Sucess using eval instead of exec
    signal = eval(signalexpr.tolist().split(b'=')[1])
    time_unit_in_seconds = timeunitexpr.tolist()
    timebase = time_unit_in_seconds * eval(timebaseexpr.tolist().split(b'=')[1])
    retdic = {"signal":signal, "timebase":timebase, "parent_element":
              dic['parent_element'], "params": dic['params'].tolist()}
    return(retdic)

if sys.version<'3,':
    from pyfusion.data.save_compress import newload
else:
    newload = newloadv3

def try_fetch_local(input_data, bare_chan):
    """ return data if in the local cache, otherwise None
    doesn't work for single channel HJ data.
    sgn (not gain) be only be used at the single channel base/fetch level
    """
    for each_path in pyfusion.config.get('global', 'localdatapath').split('+'):
        # check for multi value shot, e.g. utc bounds for W7-X data
        shot = input_data.shot
        if isinstance(shot, (tuple, list, ndarray)):
            shot_str = '{s0}_{s1}'.format(s0=shot[0], s1=shot[1])
        else:
            shot_str = str(shot)
        input_data.localname = os.path.join(each_path, '{shot}_{bc}.npz'
                                          .format(shot=shot_str, bc=bare_chan))
        # original - data_filename %filename_dict)
        files_exist = os.path.exists(input_data.localname)
        debug_(pyfusion.DEBUG, 3, key='try_local_fetch')
        if files_exist: break

    if not files_exist:
        return None

    signal_dict = newload(input_data.localname)
    
#    coords = get_coords_for_channel(**input_data.__dict__)
    ch = Channel(bare_chan,  Coords('dummy', (0,0,0)))
    output_data = TimeseriesData(timebase=Timebase(signal_dict['timebase']),
                             signal=Signal(signal_dict['signal']), channels=ch)
    # bdb - used "fetcher" instead of "self" in the "direct from LHD data" version
    #  when using saved files, should use the name - not input_data.config_name
    #  it WAS the config_name coming from the raw format.
    output_data.config_name = bare_chan
    # would be nice to get to the gain here - but how - maybe setup will get it
    output_data.meta.update({'shot':input_data.shot})
    if 'params' in signal_dict: 
        output_data.params = signal_dict['params']
        if 'utc' in signal_dict['params']:
            output_data.utc =  signal_dict['params'].get('utc',None)
    else:
        # yes, it seems like duplication, but no
        output_data.utc = None
    return(output_data)



class BaseAcquisition(object):
    """Base class for datasystem specific acquisition classes.

    :param   config_name:  name   of  acquisition   as  specified in\
    configuration file.

    On  instantiation,  the pyfusion  configuration  is  searched for  a
    ``[Acquisition:config_name]``   section.    The   contents  of   the
    configuration  section are  loaded into  the object  namespace.  For
    example, a configuration section::

      [Acquisition:my_custom_acq]
      acq_class = pyfusion.acquisition.base.BaseAcquisition
      server = my.dataserver.com
 
    will result in the following behaviour::

     >>> from pyfusion.acquisition.base import BaseAcquisition
     >>> my_acq = BaseAcquisition('my_custom_acq')
     >>> print(my_acq.server)
     my.dataserver.com

    The configuration entries can be overridden with keyword arguments::

     >>> my_other_acq = BaseAcquisition('my_custom_acq', server='your.data.net')
     >>> print(my_other_acq.server)
     your.data.net

    """
    def __init__(self, config_name=None, **kwargs):
        if config_name is not None:
            self.__dict__.update(get_config_as_dict('Acquisition', config_name))
        self.__dict__.update(kwargs)

    def getdata(self, shot, config_name=None, **kwargs):
        """Get the data and return prescribed subclass of BaseData.
        
        :param shot: shot number
        :param config_name: ?? bdb name of a fetcher class in the configuration file
        :returns: an instance of a subclass of \
        :py:class:`~pyfusion.data.base.BaseData` or \
        :py:class:`~pyfusion.data.base.BaseDataSet`

        This method needs to know which  data fetcher class to use, if a
        config_name      argument     is      supplied      then     the
        ``[Diagnostic:config_name]``   section   must   exist   in   the
        configuration   file  and   contain  a   ``data_fetcher``  class
        specification, for example::

         [Diagnostic:H1_mirnov_array_1_coil_1]
         data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
         mds_path = \h1data::top.operations.mirnov:a14_14:input_1
         coords_cylindrical = 1.114, 0.7732, 0.355
         coord_transform = H1_mirnov

        If a ``data_fetcher`` keyword argument is supplied, it overrides
        the configuration file specification.

        The  fetcher  class  is  instantiated,  including  any  supplied
        keyword arguments, and the result of the ``fetch`` method of the
        fetcher class is returned.
        """
        from pyfusion import config
        # if there is a data_fetcher arg, use that, otherwise get from config
        if 'data_fetcher' in kwargs:
            fetcher_class_name = kwargs['data_fetcher']
        else:
            fetcher_class_name = config.pf_get('Diagnostic',
                                               config_name,
                                               'data_fetcher')
        fetcher_class = import_from_str(fetcher_class_name)
        ## Problem:  no check to see if it is a diag of the right device!??
        d = fetcher_class(self, shot,
                             config_name=config_name, **kwargs).fetch()
        d.history += "\n:: shot: {s} :: config: {c}".format(s=shot, c=config_name)

        return d
        
class BaseDataFetcher(object):
    """Base  class  providing  interface   for  fetching  data  from  an
    experimental database.
    
    :param acq: in instance of a subclass of :py:class:`BaseAcquisition`
    :param shot: shot number
    :param config_name: name of a Diagnostic configuration section.
    
    It is expected that subclasses of BaseDataFetcher will be called via
    the :py:meth:`~BaseAcquisition.getdata` method, which calls the data
    fetcher's :py:meth:`fetch` method.
    """
    def __init__(self, acq, shot, config_name=None, **kwargs):
        self.shot = shot
        self.acq = acq
        #bdb?? add device name here, so can prepend to Diagnostic
        # e.g. LHD_Diagnostic - avoids ambiguity
        debug_(pyfusion.DEBUG,5,key='device_name')
        if config_name is not None:
            self.__dict__.update(get_config_as_dict('Diagnostic', config_name))
            if pyfusion.VERBOSE>3: print(get_config_as_dict('Diagnostic', config_name))
        self.__dict__.update(kwargs)
        self.config_name=config_name
#        print('BaseDFinit',config_name,self.__dict__.keys())

    def setup(self):
        """Called by :py:meth:`fetch` before retrieving the data."""
        pass

    def do_fetch(self):
        """Actually fetches  the data, using  the environment set  up by
        :py:meth:`setup`

        :returns: an instance of a subclass of \
        :py:class:`~pyfusion.data.base.BaseData` or \
        :py:class:`~pyfusion.data.base.BaseDataSet`

        Although :py:meth:`BaseDataFetcher.do_fetch` does not return any
        data object itself, it is expected that a `do_fetch()` method on
        a subclass of :py:class:`BaseDataFetcher` will.
        """
        pass

    def pulldown(self):
        """Called by :py:meth:`fetch` after retrieving the data."""
        pass

    def error_info(self, step=None):
        """ return specific information about error to aid interpretation - e.g for mds, path
        The dummy return should be replaced in the specific routines
        """
        return('(further info not provided by %s)' % (self.acq.acq_class))

    def fetch(self):
        """Always use  this to fetch the data,  so that :py:meth:`setup`
        and  :py:meth:`pulldown` are  used to  setup and  pull  down the
        environmet used by :py:meth:`do_fetch`.
        
        :returns: the instance of a subclass of \
        :py:class:`~pyfusion.data.base.BaseData` or \
        :py:class:`~pyfusion.data.base.BaseDataSet` returned by \
        :py:meth:`do_fetch`
        """        
        if pyfusion.DEBUG>2:
            exception = ()  # defeat the try/except
        else: exception = Exception
        sgn = 1
        chan = self.config_name
        if chan[0]=='-': #sgn = -sgn
            raise ValueError('Channel {chan} should not have sign at this point'
                             .format(chan=chan))
        #bare_chan = (chan.split('-'))[-1]
        # use its gain, or if it doesn't have one, its acq gain.
        if pyfusion.RAW == 0:
            gain_units = "1 arb"  # default to gain 1, no units
            if hasattr(self,'gain'):
                gain_units = self.gain
            elif hasattr(self.acq, 'gain'):
                gain_units = self.acq.gain
        else:
            gain_units = "1 raw"

        sgn *= float(gain_units.split()[0])
        if pyfusion.VERBOSE > 0: print('Gain factor', sgn, self.config_name)

        tmp_data = try_fetch_local(self, chan)  # is there a local copy?
        if tmp_data is None:
            method = 'specific'
            try:
                self.setup()
            except exception as details:
                traceback.print_exc()
                raise LookupError("{inf}\n{details}".format(inf=self.error_info(step='setup'),details=details))
            try:
                data = self.do_fetch()
            except exception as details:   # put None here to show exceptions.
                                           # then replace with Exception once
                                           # "error_info" is working well

                # this is to provide traceback from deep in a call stack
                # the normal traceback doesn't see past the base.py into whichever do_fetch
                #  Avoid printing exception stuff unless we want a little verbosity
                if pyfusion.VERBOSE>0:
                    #  this simple method doesn't work, as it only has info after 
                    #  getting to the prompt
                    if  hasattr(sys, "last_type"):
                        traceback.print_last()
                    else: 
                        print('sys has not recorded any exception - needs to be at prompt?')

                    # this one DOES work.
                    print(sys.exc_info())
                    (extype, ex, tb) = sys.exc_info()
                    for tbk in traceback.extract_tb(tb):
                        print("Line {0}: {1}, {2}".format(tbk[1],tbk[0],tbk[2:]))

                raise LookupError("{inf}\n{details}{CLASS}".format(inf=self.error_info(step='setup'),
                                                            details=details,CLASS=details.__class__))
        else:
            data = tmp_data
            method = 'local_npz'

        data.meta.update({'shot':self.shot})
        data.signal = sgn * data.signal
        if not hasattr(data,'utc'):
            data.utc = None
        # Coords shouldn't be fetched for BaseData (they are required
        # for TimeSeries)
        #data.coords.load_from_config(**self.__dict__)
        if pyfusion.VERBOSE>0: 
            print("base.py: data.config_name", data.config_name)
        data.channels.config_name=data.config_name
        if len(gain_units.split()) > 1:
            data.channels.units = gain_units.split()[-1]
        else:
            data.channels.units = data.units if hasattr(data,'units') else ' '
        if method == 'specific':  # don't pull down if we didn't setup
            self.pulldown()
        return data

class MultiChannelFetcher(BaseDataFetcher):
    """Fetch data from a diagnostic with multiple timeseries channels.

    This fetcher requres a multichannel configuration section such as::

     [Diagnostic:H1_mirnov_array_1]
     data_fetcher = pyfusion.acquisition.base.MultiChannelFetcher
     channel_1 = H1_mirnov_array_1_coil_1
     channel_2 = H1_mirnov_array_1_coil_2
     channel_3 = H1_mirnov_array_1_coil_3
     channel_4 = H1_mirnov_array_1_coil_4

    The channel  names must be  `channel\_` followed by an  integer, and
    the channel  values must correspond to  other configuration sections
    (for        example       ``[Diagnostic:H1_mirnov_array_1_coil_1]``,
    ``[Diagnostic:H1_mirnov_array_1_coil_1]``, etc)  which each return a
    single               channel               instance               of
    :py:class:`~pyfusion.data.timeseries.TimeseriesData`.
    """
    def ordered_channel_names(self):
        """Get an ordered list of the channel names in the diagnostic

        :rtype: list
        """
        channel_list = []
        for k in self.__dict__.keys():
            if k.startswith('channel_'):
                channel_list.append(
                    [int(k.split('channel_')[1]), self.__dict__[k]]
                    )
        channel_list.sort()
        return [i[1] for i in channel_list]
    
    def fetch(self):
        """Fetch each  channel and combine into  a multichannel instance
        of :py:class:`~pyfusion.data.timeseries.TimeseriesData`.

        :rtype: :py:class:`~pyfusion.data.timeseries.TimeseriesData`
        """
 
        ## initially, assume only single channel signals
        # this base debug breakpoint will apply to all flavours of acquisition
        debug_(pyfusion.DEBUG, level=3, key='entry_base_multi_fetch')
        ordered_channel_names = self.ordered_channel_names()
        data_list = []
        channels = ChannelList()  # empty I guess
        common_tb = None  # common_timebase
        meta_dict={}

        group_utc = None  # assume no utc, will be replaced by channels
        # t_min, t_max seem obsolete, and should be done in single chan fetch
        if hasattr(self, 't_min') and hasattr(self, 't_max'):
            t_range = [float(self.t_min), float(self.t_max)]
        else:
            t_range = []
        for chan in ordered_channel_names:
            sgn = 1
            if chan[0]=='-': sgn = -sgn
            bare_chan = (chan.split('-'))[-1]
            ch_data = self.acq.getdata(self.shot, bare_chan)
            if len(t_range) == 2:
                ch_data = ch_data.reduce_time(t_range)

            channels.append(ch_data.channels)
            # two tricky things here - tmp.data.channels only gets one channel hhere
            # Config_name for a channel is attached to the multi part -
            # We need to move it to the particular channel 
            # Was  channels[-1].config_name = chan
            # 2013 - default to something if config_name not defined
            if pyfusion.VERBOSE>0:
                print("base:multi ch_data.config_name", ch_data.config_name)
            if hasattr(ch_data,'config_name'):
                channels[-1].config_name = ch_data.config_name                
            else:
                channels[-1].config_name = 'fix_me'
            meta_dict.update(ch_data.meta)
            # tack on the data utc
            ch_data.signal.utc = ch_data.utc

            # Make a common timebase and do some basic checks.
            if common_tb is None:
                common_tb = ch_data.timebase
                if hasattr(ch_data,'utc'):
                    group_utc = ch_data.utc
                    tb_chan = ch_data.channels

                # for the first, append the whole signal
                data_list.append(sgn * ch_data.signal)
            else:
                if hasattr(self, 'skip_timebase_check') and self.skip_timebase_check == 'True':
                    # append regardless, but will have to go back over
                    # later to check length cf common_tb
                    if pyfusion.VERBOSE > 0: print('utcs: ******',ch_data.utc[0],group_utc[0])
                    if ch_data.utc[0] != group_utc[0]:
                        # should pad this channel out with nans - for now report an error.
                        print('********\n********traces {tbch} and {chcnf} have different start times by {dts:.2g} s'
                              .format(tbch = tb_chan.config_name, chcnf = ch_data.config_name,
                                      dts = (ch_data.utc[0] - group_utc[0])/1e9))

                    if len(ch_data.signal)<len(common_tb):
                        common_tb = ch_data.timebase
                        tb_chan = ch_data.channels
                    data_list.append(ch_data.signal[0:len(common_tb)])
                else:
                    try:
                        assert_array_almost_equal(common_tb, ch_data.timebase)
                        data_list.append(ch_data.signal)
                    except:
                        print('####  matching error in {c} - perhaps timebase not the same as the previous channel'.format(c=ch_data.config_name))
                        raise 

        if hasattr(self, 'skip_timebase_check') and self.skip_timebase_check == 'True':
            #  Messy - if we ignored timebase checks, then we have to check
            #  length and cut to size, otherwise it will be stored as a signal (and 
            #  we will end up with a signal of signals)
            #  This code may look unpythonic, but it avoids a copy()
            #  and may be faster than for sig in data_list....
            ltb = len(common_tb)
            for i in range(len(data_list)):
                if len(data_list[i]) > ltb:
                    # this is a replacement.
                    data_list.insert(i,data_list.pop(i)[0:ltb])

        signal = Signal(data_list)
        print(shape(signal))

        output_data = TimeseriesData(signal=signal, timebase=common_tb,
                                     channels=channels)
        #output_data.meta.update({'shot':self.shot})
        output_data.meta.update(meta_dict)
        #if not hasattr(output_data,'utc'):
        #    output_data.utc = None  # probably should try to get it
        #                           # from a channel - but how?
        output_data.utc = group_utc   # should check that all channels have same utc
        debug_(pyfusion.DEBUG, level=2, key='return_base_multi_fetch')        
        return output_data

