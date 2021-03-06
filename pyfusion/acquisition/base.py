"""Base classes for  pyfusion data acquisition. It is  expected that the
base classes  in this module will  not be called  explicitly, but rather
inherited by subclasses in the acquisition sub-packages.

"""
from __future__ import print_function
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import ndarray, shape
from time import time as seconds
from pyfusion import config
from pyfusion.conf.utils import import_setting, kwarg_config_handler, \
     get_config_as_dict, import_from_str, NoSectionError

from pyfusion.data.timeseries import Signal, Timebase, TimeseriesData
from pyfusion.data.base import Coords, Channel, ChannelList, get_coords_for_channel
from pyfusion.debug_ import debug_
import traceback
import sys, os
import pyfusion  # only needed for .VERBOSE and .DEBUG
from pyfusion.acquisition.W7X.get_shot_info import get_shot_utc
from pyfusion.acquisition.W7X import interpolate_corrupted_W7X_timebase
from distutils.version import LooseVersion

CONTINUE_PAST_EXCEPTION = 2  # level of pyfusion.DEBUG below which exceptions in fetch are continued over

def newloadv3(filename, verbose=1):
    """ This the the version that works in python 3, but can't handle Nans yet
    because exec() is not supported in Py3 and the code below so far only 
    chooses the expressions to be exec'ed apart simplistically.
    Intended to replace load() in numpy
    Its counterpart is data/savez_compress.py
    """
    from numpy import load as loadz
    from numpy import cumsum, array
    dic=loadz(filename)
#    if dic['version'] != None:
#    if len((dic.files=='version').nonzero())>0:
    if len(dic.files)>3:
        if verbose>2: print ("local v%d " % (dic['version']),end='')
    else: 
        if verbose>2: print("local v0: simple ", end='')
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
    print(timebaseexpr)
    # !! py3 - this was suppressed in py3, by luru, but I think we need it - bdb
    #print(timebaseexpr.tolist().split(b'=')[1])
    timebase = time_unit_in_seconds * eval(timebaseexpr.tolist().split(b'\n')[0].split(b'=')[1])
    retdic = {"signal":signal, "timebase":timebase, "parent_element":
              dic['parent_element'], "params": dic['params'].tolist()}
    return(retdic)

if sys.version<'3,':
    from pyfusion.data.save_compress import newload
else:
    newload = newloadv3

def try_fetch_local(input_data, bare_chan, time_range=None):
    """ return data if in the local cache, otherwise None
    doesn't work for single channel HJ data.
    sgn (not gain) be only be used at the single channel base/fetch level
    shot can be a utc, but only if the stored data is of that form.
    """
    for each_path in pyfusion.config.get('global', 'localdatapath').split('+'):
        # check for multi value shot number, e.g. utc bounds for W7-X data
        shot = input_data.shot
        # MDSplus style path to access sorted files into folders by shot
        path, patt = os.path.split(each_path)
        #  print(patt)
        # Detect a a subdir code based on date/shot - half the chars are ~
        if len(patt) == 2*len(patt.replace('~','')): 
            subdir = ''
            # Use the YYYYMMDD part if two components
            strshot = str(shot[0]) if len(np.shape(shot))>0 else str(shot)
            # print(strshot, patt, str(shot))
            if len(np.shape(shot)) == 0 and int(strshot[2:4]) <16:
                print("*******Warning {strshot} seems like an MDSplus shot or MDS test shot in YYMMDD form - see line 255"
                      .format(strshot=strshot))
            # reverse the order of both the pattern and the shot so 'a' posn is 0th char
            revshot = strshot[::-1]
            for i,ch in enumerate(patt):
                if (i%2) == 0: 
                    if ch != '~':
                        raise LookupError("Can't parse {d} as a MDS style subdir"
                                          .format(d=patt))
                    continue
                if (ord(ch) - ord('a')) < len(revshot):
                    subdir += revshot[ord(ch) - ord('a')]
                else:
                    if pyfusion.VERBOSE>0:
                        print('********** Are we working with MDSplus W7M test shot??? - \n need a W7M test shot to check if this is really and error', shot)

        else:
            subdir = patt
        debug_(pyfusion.DEBUG, 4, key='MDSPlus style subdir ~path', msg=each_path)
        each_path = os.path.join(path, subdir)
        # implement shot[1] < 1 for MDSplus test shots - the only
        # sacrifice is that we can't use shot[1] 0 to be the latest
        # but we could still use shot =[0,0] to be the latest.

        # for now, allow - in file names - later we could replace '-' with 'M'
        if isinstance(shot, (tuple, list, ndarray)):
            shot_str = '{s0}_{s1}'.format(s0=shot[0], s1=shot[1])
        else:
            shot_str = str(shot)
        input_data.localname = os.path.join(each_path, '{shot}_{bc}.npz'
                                          .format(shot=shot_str, bc=bare_chan))
        # original - data_filename %filename_dict)
        files_exist = os.path.exists(input_data.localname)
        if pyfusion.VERBOSE>2: print('search', each_path, 'for', input_data.localname, ['No!','Found!'][files_exist])
        debug_(pyfusion.DEBUG, 3, key='try_local_fetch')
        if files_exist: 
            intmp = np.any([st in input_data.localname.lower() for st in 
                            ['tmp', 'temp']])  # add anything you wish to warn about
            if pyfusion.VERBOSE>0 or intmp:
                if intmp: 
                    pyfusion.logging.warning('Using {f} in temporary directory!'
                                  .format(f=input_data.localname))
                print('found local data in {f}'. format(f=input_data.localname))
            break

    if not files_exist:
        return None

    signal_dict = newload(input_data.localname)
    """ These W7X-specific lines are here to deal with npz saved data
    See the W7X fetch.py for more info
    Examples of difficult shots.
    'W7X_L53_LP01_U' shot_number=[20160309,13]  0.8.2b writes all zeros (when 
        read by newload 0.9.92O clean,on both) even though the rawdim is OK up to 350,000
    L57  doesn't have this problem in 0.8.2b - written the same date! (but diff dimraw very fragmented)
    """
    if 'params' not in signal_dict:
        signal_dict.update(dict(params=()))
    if 'params' in signal_dict and 'name' in signal_dict['params'] and 'W7X_L5' in signal_dict['params']['name']:
        if  LooseVersion(signal_dict['params']['pyfusion_version']) < LooseVersion('0.6.8b'):
            raise ValueError('probe assignments in error LP11-22 in {fn}'
                             .format(fn=input_data.localname))
        if np.nanmax(signal_dict['timebase']) == 0:
            pyfusion.logging.warning("======== all 0's: making a fake timebase for {fn}"
                                     .format(fn=input_data.localname))
            signal_dict['timebase'] = 2e-6*np.cumsum(1.0 + 0*signal_dict['signal'])

        if np.diff(signal_dict['timebase'])[0] == 0: # first two are the same
            signal_dict['timebase'] = interpolate_corrupted_W7X_timebase(signal_dict)

#        if 'req_f_u' not in signal_dict['params']:  # req_f_u was initially called seg_f_u (incorrectly)
#            signal_dict['params']['req_f_u'] = signal_dict['params']['shot_f']
        if 'utc_0' in signal_dict['params']:
            true_start = (signal_dict['params']['data_utc'][0] - signal_dict['params']['utc_0'])/1e9
        else:
            true_start = (signal_dict['params']['data_utc'][0] - signal_dict['params']['shot_f'])/1e9  - 61.0

        print('\n base:.py ****True_start of timebase', true_start)
        delta_t = true_start - signal_dict['timebase'][0] 
        # used to want timebase starting at zero - changed in 2020 - probably wrong for a long time.
        
        if np.abs(delta_t) > 1e-6:
            if  LooseVersion(signal_dict['params']['pyfusion_version']) >= LooseVersion('0.9.94O'):  #note the O!
                #  not accessible here - if signal_dict['version'] >= 106:
                print('not adjusting')
            else:
                # Ideally the timebase in seconds should be 0 secs at t1 (diag trigger)
                print("=== correcting timebase which has not been adjusted to t=0 - discrepancy = {dt:.4g}"
                      .format(dt=delta_t))
                # Note that the relevant version is at the time of saving.
                signal_dict['timebase'] = signal_dict['timebase'] - delta_t
                

    coords = get_coords_for_channel(**input_data.__dict__)
    #ch = Channel(bare_chan,  Coords('dummy', (0,0,0)))
    ch = Channel(bare_chan,  coords)
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
            output_data.params = dict(comment = 'old npz file has no params')

    # If we are sub-selecting, the utc should be adjusted.
    if time_range is not None:
        origbnds = (output_data.timebase[[0,-1]]/1e-9).astype(np.float) #  can't use min max as we want the ends
        output_data = output_data.reduce_time(time_range)  # because reduce time can only copy now.
        # ***** this output_data.utc calc only works for non-nan timebase start and ends,
        #       and only if the original output_data.utc is correct
        newbnds = (output_data.timebase[[0,-1]]/1e-9).astype(np.float)
        output_data.utc = [output_data.utc[i] + (newbnds[i] - origbnds[i]).round(0) for i in range(2)]
        print('output_data.utc', output_data.utc, newbnds, origbnds)

    oldsrc =  ', originally from ' + output_data.params['source'] if hasattr(output_data, 'params') and 'source' in output_data.params else ''
    output_data.params.update(dict(source='from npz cache' + oldsrc))
    return(output_data)

def update_with_valid_config(fetcher):
    """
    Look for the first valid shot range for this fetcher working backwards
    and update the fetcher with its dictionary info
       see config/'Valid Shots' in the reference docs
    This code was extracted into a function
    It is called very early in the base.py code - the  _init_() phase
    """
    def valid_for_shot(fetcher):
        """ Determine if this fetcher's diag definition or modified diag is valid 
        for this shot
        """

        def check_to_clause(shot, k, dic):
            """ check for leftover instances of to=[date, shot]
            where shot is 0 - this is very likely to be a mistake
            as there is no shot before 0
            """
            if ('_to' in k and isinstance(shot,(list, tuple, ndarray)) 
                and dic[k][1] == 0):
                print('********  Warning - valid shot of 0 in to clause?')


        if hasattr(fetcher,'valid_shots'):
            valid_shots = fetcher.valid_shots
        elif hasattr(fetcher.acq, 'valid_shots'):
            valid_shots = fetcher.acq.valid_shots
        else:
            valid_shots = None
        # another example of inheritance via pyfusion.cfg 
        #   - need to formalise this, extract the code to a function?
        is_valid = True

        # check for an MDSplus W7M test shot 18... - normal W7M is 2018.... - need to find command lines from ipp
        # Starting with 18 instead of 2018 allows pyfusion to distinguish test shots - on the W7X net,
        # perhaps these are distinguished by using a different tree, because the shot number may be duplicated
        # Examplee:  run pyfusion/examples/plot_signals.py dev_name=W7M diag_name=W7M_BRIDGE_V1 shot_number=[180907,9]
        if np.shape(fetcher.shot) != () and fetcher.shot[0] < 990000:  
            valid_shots = None # don't check as the check will find no shot at the moment
            pyfusion.utils.warn('ignoring valid_since data assumed for MDS test {sh}'
                                     .format(sh=str(fetcher.shot)))

        if valid_shots is not None:
            shot_or_utc = fetcher.shot
            # this 15 line code block is W7X specific - how to remove to W7X?
            if np.isscalar(shot_or_utc):
                compfun = int
            else:
                compfun = tuple
            valid_dict = eval('dict({ps})'.format(ps=valid_shots))
            for k in valid_dict:
                root = k.replace('_from','').replace('_to','')
                if '_' + root in fetcher.config_name:
                    if pyfusion.VERBOSE>1: print('find_valid_for_shot: key={k}, root={r} shot={s} valid={v}'
                                                 .format(k=k, r=root, s=fetcher.shot, v=valid_dict[k]))
                    check_to_clause(shot_or_utc, k, valid_dict)
                    # need to be both tuples or both lists for comparison to work
                    if (('_from' in k and compfun(get_shot_utc(shot_or_utc)) < compfun(get_shot_utc(valid_dict[k])))
                        or ('_to' in k and compfun(get_shot_utc(shot_or_utc)) > compfun(get_shot_utc(valid_dict[k])))):
                        is_valid = False
        debug_(pyfusion.DEBUG, 2, 'valid_shots')
        return(is_valid)

    # now the actual code       
    
    devshort = fetcher.config_name.split('_')[0]  # assume the device short name is first, before the _
    #  config_dict = {}
    for Mod in ['M1', 'M2', 'M3', 'M4', 'M5']:
        if valid_for_shot(fetcher):  
            break
        else:
            # replace W7X_ with W7XM1_  etc
            fetcher.__dict__.update(get_config_as_dict('Diagnostic', fetcher.config_name.replace(devshort, devshort+Mod)))
            if pyfusion.VERBOSE>-1: print('### >> Loading config from {m} -> {d}\n'.format(m=Mod, d=fetcher.__dict__))

    else:  # here if we complete the for loop - it is an error
        raise LookupError(config_name)

    return fetcher.__dict__

    # fetcher.config_name=config_name
    # print('BaseDFinit',config_name,fetcher.__dict__.keys())


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

    # This is also used by the multi,  as it is not overridden, but explicitly
    #  in user code in a call to getdata.  The multi class fetch then calls getdata
    #  again from within fetch but as a single diag class.
    # Contrast this with fetch - which are separate funtions for multi and single.
    def getdata(self, shot, config_name=None, interp={}, quiet=0, contin=False, exceptions=None, time_range=None, **kwargs):
        """Get the data and return prescribed subclass of BaseData.
        
        :param shot: shot number
        :param config_name: ?? bdb name of a fetcher class in the configuration file
        :param interp: dictionary specifying interpolation method, grid
            methods are 'linear', 'linear_minmax' (W7X,H1) - NOT YET IMPL..
        :param exceptions - a list of exceptions to catch - None will 
            default to (LookupError) or () if pyfusion.DEBUG>2
        :returns: an instance of a subclass of \
        :py:class:`~pyfusion.data.base.BaseData` or \
        :py:class:`~pyfusion.data.base.BaseDataSet`

        contin [False] - if True, continue with None for data
        quiet [0] Quiet counters the effect of pyfusion.VERBOSE for this routine - net effect is VERBOSE-quiet

        Call chain is:
        acq.getdata -> fetcher_class.fetch() chosen according to kwargs or diagn .cfg file
        This always goes via base.fetch() because it isn't overridden in specific fetch.py
        base.fetch then calls
        -> fetcher_class.setup(),  fetcher_class do_fetch() and pull_down

        An exception is the Multichannel fetcher_class, which is dealt with 
        inside base.py.  The same getdata function is used 
        (as Multi doesn't override getdata) which then calls the multi channel's
        fetch function, which calls the getdata in the base.py 
        as above on each channel.

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
        if pyfusion.VERBOSE >= 0:
            print('fetcher', shot, end=', ')
        if shot is None:
            raise ValueError('Attempt to get data for shot = None')
        if time_range is not None:
            self.time_range = time_range

        if exceptions is None:
            exceptions = (LookupError) if pyfusion.DEBUG < CONTINUE_PAST_EXCEPTION else ()  # also in 428

        # if there is a data_fetcher arg, use that, otherwise get from config
        if config_name.lower() == 'none':
            return(None)
        try:
            if 'data_fetcher' in kwargs:
                fetcher_class_name = kwargs['data_fetcher']
            else:
                fetcher_class_name = config.pf_get('Diagnostic',
                                                   config_name,
                                                   'data_fetcher')
        except NoSectionError as reason:  # suggest 'similar' names from sections
            our_sections = config.sections()
            from difflib import SequenceMatcher
            matches = np.argsort([SequenceMatcher(None, col, config_name).ratio() for col in our_sections])
            print(reason.__repr__())
            print('\nInstead of {v} try {c}'
                  .format(v=config_name, c=[our_sections[m] for m in matches[::-1][:10]]))
            if pyfusion.VERBOSE>0:
                raise
            else:
                sys.exit()
            
        fetcher_class = import_from_str(fetcher_class_name)
        if pyfusion.VERBOSE > 2:
            print('fetcher class is', fetcher_class_name)
        ## Problem:  no check to see if it is a diag of the right device!??
        # enable stopping here on error to allow traceback if DEBUG>2
        # there is similar code elsewhere - check if is duplication
        if pyfusion.VERBOSE > 2:
            print('shot {0}, fetcher_class_name {1} '.format(shot, fetcher_class_name))
            
        fetcher_class.contin = contin
        fetcher_class.time_range = time_range
        if pyfusion.VERBOSE > 0:
            print('fetcher time range = ', time_range, end=', ')
        try:
            d = fetcher_class(self, shot, interp=interp,
                              config_name=config_name, **kwargs).fetch()
        except exceptions as reason:
            debug_(pyfusion.DEBUG, 1, key='failed_fetch')
            if contin:
                if pyfusion.VERBOSE-quiet >= 0:
                    pyfusion.utils.warn('Exception: ' + str(reason))
                    print('Exception: ' + str(reason))
                return None
            else:
                raise
        if d is not None:
            d.history += "\n:: shot: {s} :: config: {c}".format(s=shot, c=config_name)

        if pyfusion.NSAMPLES != 0:
            d.timebase = np.clip(d.timebase, -10, 100) # should really do it to shot_t - shot_t
            
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
        """ Accepts scalar(LHD/H-1) or tuple/list (MIT/W7X) shot numbers 
        The tuple may be date,progid or from,upto in utc ns
        Testing: try these three 
           run pyfusion/examples/plot_signals shot_number=97661 diag_name='H1_CliveProbeI' dev_name='H1Local'
           run -i pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name='W7X_L53_LP01_I' shot_number=[20160310,40]
           run -i pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name='W7X_L53_LP01_I' shot_number=data.utc
        # if the W7X server is not accessible, the last should error looking for the W7X server
        """
        self.shot = shot
        self.acq = acq
        self.no_cache = False  # this allows getData to request that cached data is NOT used - eg for saving local
        #bdb?? add device name here, so can prepend to Diagnostic
        # e.g. LHD_Diagnostic - avoids ambiguity
        debug_(pyfusion.DEBUG,5,key='device_name')
        if config_name is not None: 
            self.__dict__.update(get_config_as_dict('Diagnostic', config_name))
            self.config_name = config_name
            self.__dict__.update(update_with_valid_config(self))
        else:
            pass # should do something here??
        self.__dict__.update(kwargs)


    def setup(self):
        """Called by :py:meth:`fetch` before retrieving the data.
        setup() ideally does sufficient of the preliminaries for fetch
        so that their is enough information in self. for a useful error 
        report if fetch fails, while avoiding exceptions wherever 
        possible (during setup()).
        """
        pass  # base:setup should always be overridden

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
        The dummy return should be replaced in the specific routines, or
        the info passed through self.errmsg
        """
        pyfusion.warn('acquisition-specific error_info not implemented for ', self.name)
        if hasattr(self, 'errmsg'):
            return(self.errmsg)
        else:
            return('(further info not provided by %s)' % (self.acq.acq_class))

    def fetch(self):
        """Always use  this to fetch the data,  so that :py:meth:`setup`
        and  :py:meth:`pulldown` are  used to  setup and  pull  down the
        environment used by :py:meth:`do_fetch`.
        
        :returns: the instance of a subclass of \
        :py:class:`~pyfusion.data.base.BaseData` or \
        :py:class:`~pyfusion.data.base.BaseDataSet` returned by \
        :py:meth:`do_fetch`
        """        
        if pyfusion.DBG() < CONTINUE_PAST_EXCEPTION:   # need to control this also at 297
            exception = Exception                     # defeat the try/except
        else: exception = ()
        oldsgn = 1  # this wasn't relevant - only relevant for multi diags
        cal_info = {}
        chan = self.config_name
        if chan[0]=='-': #sgn = -sgn
            raise ValueError('Channel {chan} should not have sign at this point'
                             .format(chan=chan))
        #bare_chan = (chan.split('-'))[-1]
        # use its gain, or if it doesn't have one, its acq gain.
        
        if hasattr(self,'minerva_name'):
            from pyfusion.acquisition.W7X.get_url_parms import get_minerva_parms
            debug_(pyfusion.DEBUG, level=2, key='MinervaName')
            self, cal_info = get_minerva_parms(self)
            # add the power supply in
            params_dict = eval('dict('+self.params+')')
            if 'powerSupply' in params_dict:
                self.vsweep = 'W7X_PSUP{n}_U'.format(n=params_dict['powerSupply'][-1])
                
        if pyfusion.RAW == 0:
            #  Implement a single argument function - no spaces allowed,
            #  data is self_signal, will ignore gain
            # e.g. expr = -10/self_signal
            ##  Note that it only is applied when pyfusion.RAW is not in effect
            #  (so it is not applied during save_local, which sets RAW at present 
            debug_(pyfusion.DEBUG, level=2, key='expr')
            if hasattr(self, 'expr'):
                if len(self.expr.split()) != 2:
                    raise ValueError('expr must have units and otherwise, no spaces')
                expr, units = self.expr.split()
                gain = None
            else:
                expr = None
                gain_units = "1 arb"  # default to gain 1, no units
                if hasattr(self,'gain'):
                    gain_units = self.gain
                elif hasattr(self.acq, 'gain'):
                    gain_units = self.acq.gain
                gain = float(gain_units.split()[0])
                if len(gain_units.split()) > 1:
                    units = gain_units.split()[-1]
                else:
                    units = ' ' #data.channels.units = data.units if hasattr(data,'units') else ' '
                if hasattr(self, 'fixup_gain'):
                    print('************ fixup_gain ***************')
                    gain *= float(self.fixup_gain)
        else:  #  pyfusion.RAW is set
            gain_units = '1 raw'
            gain, units = gain_units.split()
            gain = float(gain)
            expr = None

        if pyfusion.VERBOSE > 0:
            if expr is None:
                print('Gain factor', gain, self.config_name)
            else:
                print('Transforming using', expr, self.config_name)

        if self.no_cache or pyfusion.NSAMPLES != 0: 
            if pyfusion.VERBOSE>0:
                print('** Skipping cache search as NSAMPLES>0 or no_cache is set in caller or pyfusion.cfg')
            tmp_data = None 
        else:
            tmp_data = try_fetch_local(self, chan, time_range=self.time_range)  # If there is a local copy, get it

        debug_(pyfusion.DEBUG, 6, key='base_fetch')
        # LAST_DNS_TEST is 0 at start, positive if test has occurred and negative if suppressed
        if tmp_data is None and pyfusion.LAST_DNS_TEST >= 0:
            self.gain = gain  # not needed by fetch - just to be consistent
            method = 'specific'
            try:
                self.setup()
            except exception as details:
                print('Exception: ' + str(details))
                traceback.print_exc()
                raise LookupError("{inf}\n{details}".format(inf=self.error_info(step='setup'),details=details))
            try:
                data = self.do_fetch()
                if not hasattr(data, 'params'):
                    data.params = dict(comment='should really inherit all params from acq.fetch')
                data.params.update(dict(source=self.acq.acq_class))

            except exception as details:   # put () here to show exceptions.
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
                    if pyfusion.VERBOSE > 2:  # keep the clutter down unless verbose
                        (extype, ex, tb) = sys.exc_info()
                        for tbk in traceback.extract_tb(tb):
                            print("Line {0}: {1}, {2}".format(tbk[1],tbk[0],tbk[2:]))
                #  get what info possible
                extra = str('shot={s}, diag={d}'
                            .format(s=self.__dict__.get('shot','?'), 
                                    d=self.__dict__.get('config_name','?')))
                raise LookupError("{ex}: {inf}\n{details}{CLASS}"
                                  .format(ex=extra, inf=self.error_info(step='setup'),
                                                            details=details,CLASS=details.__class__))
        else:
            data = tmp_data
            method = 'local_npz'
            if hasattr(data, 'params') and 'DMD' in data.params:
                # important to make sure channel mapping is the same
                # c.f. changes in gain etc should be over-ridden with the latest values
                self_params = eval('dict('+self.params+')')
                if pyfusion.VERBOSE>0: print('Comparing params in base.py',self_params,'\n', data.params)
                if (self_params['DMD'] != data.params['DMD'] or
                    self_params['ch'] != data.params['ch']):
                    # below was raise Exception but really should be replace with nans?
                    pyfusion.utils.warn(('conflicting DMD/ch from npz on {s}, {d}: {dmd1}/{sc}, {dmd2}{dc}'.
                           format(s=self.shot, d=self.config_name,
                                  dmd1=self_params['DMD'], dmd2=data.params['DMD'],
                                  sc=self_params['ch'], dc=data.params['ch'])))
                else:  # DMD,ch are OK, so we can overwrite the params with the latest
                    data.params.update(self_params)
                if 'cal_date' in tmp_data.params:
                    if tmp_data.params['cal_date'] != cal_info['cal_date']:
                        pyfusion.utils.warn('Calibration out of date in npz file')
                else:
                    print("can't check cal_date")
            else: 
                if pyfusion.VERBOSE>-1 and 'W7X' in self.acq.acq_class and 'LP' in self.config_name: 
                    print("Can't check DMD on {s}, {d}".format(s=self.shot, d=self.config_name))

        if data is None:
            raise LookupError('Data not found dir {cn} on {sh} - set pyfusion.VERBOSE=3 and rerun'
                              .format(sh=str(self.shot), cn =self.config_name)) 
        if pyfusion.VERBOSE >= 0:
            print('{nsam:,d} samples'.format(nsam=len(data.signal)), end=', ')  # comma format
        if len(data.signal) == 0:
            raise LookupError('no samples in time_range of {trg} in {nm}'
                              .format(trg=str(time_range), nm=self.config_name))
        data.gain_used = gain_units if gain is not None else expr
        if hasattr(self, 'vsweep'):
            data.vsweep = self.vsweep
        data.meta.update({'shot':self.shot})
        if hasattr(data,'comment'):  # extract comment if there
            print('!data item {cn} already has comment {c}'
                  .format(cn=data.config_name, c=data.comment))
        data.comment = self.comment if hasattr(self, 'comment') else ''

        if expr is not None:  # arbitrary? expression in terms of self_signal to calc signal
            data.signal = eval(expr.replace('self_signal','data.signal'))
        else:
            data.signal = gain * data.signal
        
        data.channels.units = units
        # if full_scale is set in the config, wipe out (np.nan) points exceeding 2x
        if hasattr(self, 'full_scale'):
            wbad = np.where(np.abs(data.signal) > 2 * float(self.full_scale))[0]
            if len(wbad) > 0:
                data.signal[wbad] = np.nan

        data.cal_info = cal_info
        if not hasattr(data,'utc'):
            data.utc = None
        # Coords shouldn't be fetched for BaseData (they are required
        # for TimeSeries) - who said this?
        # data.coords.load_from_config(**self.__dict__)
        if pyfusion.VERBOSE>0: 
            print("base.py: data.config_name", data.config_name)
        data.channels.config_name=data.config_name
        if pyfusion.VERBOSE >= 0:
            print(method, end=' - ')
        if method == 'specific':  # don't pull down if we didn't setup
            self.pulldown()

        debug_(pyfusion.DEBUG, level=3, key='return_base_fetch')
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
        if len(channel_list) == 0:
            print('********* warning!! empty channel list - are there ay channel_N attributes? ')
        return [i[1] for i in channel_list]
    
    def fetch(self):
        """ Multi-channel fetcher: Fetch each  channel and combine into  a multichannel instance
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
        # t_min, t_max are obsolete, and should be done in single chan fetch

        if self.time_range is not None:
            time_range = self.time_range
        elif hasattr(self, 't_min') and hasattr(self, 't_max'):
            time_range = [float(self.t_min), float(self.t_max)]
        else:
            time_range = None
            
        params = {}  # will be added to output data, to include gain, really want Rs too
        for chan in ordered_channel_names:
            
            sgn = 1
            if chan[0]=='-': sgn = -sgn  # this allows flipping sign in the multi chan config
            bare_chan = (chan.split('-'))[-1]
            ch_data = self.acq.getdata(self.shot, bare_chan, contin=self.contin, time_range=self.time_range)
            if ch_data is None:
                print('>>>>>>>>> Skipping ', chan)
                continue
            
            if time_range is not None and len(time_range) == 2:
                ch_data = ch_data.reduce_time(time_range)

            if pyfusion.VERBOSE < 2:
                ch_data.params['diff_dimraw'] = None
            params.update({chan: dict(params=ch_data.params, gain_used=ch_data.gain_used, config_name=ch_data.config_name)})
            channels.append(ch_data.channels)
            # two tricky things here - tmp.data.channels only gets one channel here
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
            # in general we have to move the attributes we need from the channel to the channel in the multi-diag
            if hasattr(ch_data,'vsweep'):
                channels[-1].vsweep = ch_data.vsweep 
            meta_dict.update(ch_data.meta)
            # tack on the data utc
            ch_data.signal.utc = ch_data.utc

            """ *************** The common timebase issue ***************
            Note that so far, the timebase realignment code only applies to W7X
            That is that for other devices, diagnostics will only group if they 
            are on the same timebase.  This could be fixed if we could either make 
            a fake utc for those others, or introduce tb_limits to take the place 
             of the utc pair.

            See tests in W7_Limiter_Langmuir.odt under Common Timebase - 
            Two choices - 1/ don't look ahead and just take the first.
            if the next starts before or at the start, just chop off the front.
            if the next ends after or at the end, just chop off the end.
            If if starts after, pad with nans
            if it ends before, pad with nans

            or 2/ check them all first and choose the common piece

            3? compromise is to do 1/ but reorder and restart each time a 
                shortfall is found,  hopefully will only happy once. 
            """

            #  Make a common timebase and do some basic checks.
            if common_tb is None:
                print('set common tb from the first channel: ', end='')
                # bdb bug!this is a mistake unless we restrict the time range to
                #   one that suits all.  But Calling all data twice would
                #   be slow, and storing all data would use memory - see """ above
                common_tb = ch_data.timebase
                tb_chan = ch_data.channels
                if hasattr(ch_data,'utc'):
                    group_utc = ch_data.utc


                # for the first, append the whole signal
                data_list.append(sgn * ch_data.signal)
            else:  # this for each extra channel
                if hasattr(self, 'skip_timebase_check') and self.skip_timebase_check == 'True':
                    # append regardless, but will have to go back over
                    # later to check length cf common_tb
                    if pyfusion.VERBOSE > 0: print('utcs: ******',ch_data.utc[0],group_utc[0])
                    #dtclock = 2000 if self.shot[0] < 20170101 else 1/0
                    dts, counts = np.unique(np.diff(ch_data.timebase),return_counts=True)
                    dtclock = int(round(dts[counts.argmax()] * 1e9))
                    print('dtclock = {dt}ns'.format(dt=dtclock), end=',' )
                    if self.shot[0] < 20170101 and (dtclock != 2000):
                        pyfusion.logger.warning('Unexpected clock period {dtclock} on shot {s}'
                                                .format(dtclock=dtclock, s = str(self.shot)))
                    
                    dtns = (ch_data.utc[0] - group_utc[0])
                    dts = dtns/1e9
                    nsampsdiff = int(round(dtns/dtclock,0))  # difference in number of samples
                    skew = (nsampsdiff * dtclock) % dtclock
                                        
                    if ch_data.utc[0] != group_utc[0]:
                        print('*** different start times *****\n'
                              '********trace {chcnf} starts after {tbch} by {dts:.10g} s, skew = {sk}ns'
                              .format(tbch = tb_chan.config_name, chcnf = ch_data.config_name,
                                      sk = skew, dts=dts),end=' ')

                    if ch_data.utc[0] < group_utc[0]:
                        debug_(pyfusion.DEBUG, level=3, key='shortening at start')
                        ch_data.signal = ch_data.signal[-nsampsdiff:]
                        ch_data.timebase = ch_data.timebase[-nsampsdiff:]
                        ch_data.utc[0] -= nsampsdiff * int(dtclock/1e-9)  # not sure if we use this but..
                        print(' **** reducing incoming data at start')
                    elif ch_data.utc[0] > group_utc[0]:
                        print('padding incoming data at start')
                        # NOTE! leading nans in a timebase are a problem
                        # Also the signal (timeseries) probably will lose the attribute such as normalise_freq
                        # should pad this channel out with nans - for now report an error.
                        # this won't work if combining signals on a common_tb
                        # ch_data.timebase += -dts  # not sure if + or -?
                        # print(ch_data.timebase[0])
                        # works for L57_LP0701 309 13 and 0303,13
                        # also for L53_LP0701 303,13, although timebase is stairstep (using old (0.6.8.b) stored data.
                        # 
                        newlen = len(ch_data.timebase) + nsampsdiff
                        newsig = np.array(newlen * [np.nan])  # prepare an Nan signal that size
                        newtb = np.array(newlen * [np.nan])   # and timebase 
                        newsig[nsampsdiff:] = ch_data.signal  # write in the valid values
                        newtb[nsampsdiff:] = ch_data.timebase
                        debug_(pyfusion.DEBUG, level=4, key='padding start')
                        ch_data.signal = newsig               # replace
                        ch_data.timebase = newtb
                        #ch_data.timebase += dts  # not sure if + or -?
                        #"""

                    # this cuts down the length to suit the first channel
                    # but this logic assumes that the starts are the same.
                    # e.g. originally the same or fixed as above.
                    if len(ch_data.signal)<len(common_tb):
                        debug_(pyfusion.DEBUG, level=4, key='shortening')
                        common_tb = ch_data.timebase  # make the timebase shorter - the existing channels
                        tb_chan = ch_data.channels    #   will be shortened a few lines below.
                    data_list.append(sgn*ch_data.signal[0:len(common_tb)])
                else:
                    try:
                        assert_array_almost_equal(common_tb, ch_data.timebase)
                        data_list.append(sgn*ch_data.signal)
                    except:
                        print('####  matching error in {c} - perhaps timebase not the same as the previous channel'.format(c=ch_data.config_name))
                        if not self.contin:
                            raise

        if len(data_list) == 0:
            return(None)
        # This is a second time around to make sure that the lengths are
        #  the same.  Should not ne needed if the more thorough combination
        # has been done above.
        if (pyfusion.NSAMPLES != 0) or (hasattr(self, 'skip_timebase_check') and self.skip_timebase_check == 'True'):
            #  Messy - if we ignored timebase checks, then we have to check
            #  length and cut to size, otherwise it will be stored as a signal (and 
            #  we will end up with a signal of signals)
            #  This code may look unpythonic, but it avoids a copy()
            #  and may be faster than for sig in data_list....
            ltb = len(common_tb)
            for i in range(len(data_list)):
                if len(data_list[i]) > ltb:
                    debug_(pyfusion.DEBUG, level=2, key='trimming for length')
                    print('**********trimming for length')
                    # this is a replacement.
                    data_list.insert(i,data_list.pop(i)[0:ltb])

        signal = Signal(data_list)
        print(' x '.join(['{s:,}'.format(s=sh) for sh in shape(signal)]))

        output_data = TimeseriesData(signal=signal, timebase=common_tb,
                                     channels=channels)
        #output_data.meta.update({'shot':self.shot})
        output_data.meta.update(meta_dict)
        output_data.params = params
        output_data.comment = self.comment if hasattr(self, 'comment') else ''
        print('output_data.comment: [' ,output_data.comment, ']')
        #if not hasattr(output_data,'utc'):
        #    output_data.utc = None  # probably should try to get it
        #                           # from a channel - but how?
        output_data.utc = group_utc   # should check that all channels have same utc
        debug_(pyfusion.DEBUG, level=2, key='return_base_multi_fetch')        
        return output_data

