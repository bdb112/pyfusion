""" 
****  Bug! 20160824  save of a file already in local cache (e.g. to decimate) seems to square gains.

hopefully we have fixed getting wrong utc for multi channel diags!!!  *********

Saves a channel or set of channels (diagnostic) diag_name in local files
overwrite_local does an "in-place" compress if True
local_dir puts the result in that local directory. The compress_local option
doesn't check if the file is already compressed!

Examples:

# This compresses local data using newer compression methods
run examples/save_to_local.py shot_list=18993 local_dir/tmp diag_name="MP2010"
At present, only saves if compress_local is true. 

# tuning compression parameters - (this example shows very little difference
run examples/save_to_local.py shot_number=18993 'local_dir="c:/cygwin/tmp"' diag_name="mirnov_small" save_kwargs='{"delta_encode_signal":True}'

# A multi-channel diag: 
run pyfusion/examples/save_to_local.py shot_list=86507 dev_name='H1Local' diag_name='ElectronDensity15'

# simple example with a two component shot number
run pyfusion/examples/save_to_local.py shot_list='[[20160225,20]]' overwrite_local=1 dev_name='W7X' diag_name='W7X_L57_LP01_08'  local_dir='/tmp' exceptions=Exception

# realistic example with a two component shot number
run pyfusion/examples/save_to_local.py shot_list='[[20160225,s] for s in range(1,50)]'  overwrite_local=1 dev_name='W7X' diag_name='W7X_L57_LP01_08'  local_dir='/data/datamining/local_data/' exceptions=Exception

from the old pyfusion - may need to tidy up
_PYFUSION_TEST_@@local_dir=/tmp/ overwrite_local=True compress_local=1

# this example works Mar 8 2016 - need shot list and diag list to avoid problems
run  pyfusion/examples/save_to_local.py diag_name=1 diag_name="['W7X_L53_LP{nnnn}_I'.format(nnnn=nn) for nn in [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22]]" shot_list="[[20160302,s] for s in range(1,10)]"  overwrite_local=1 dev_name='W7X'  local_dir='/tmp' exceptions=Exception

# the rest of the stuff
run  pyfusion/examples/save_to_local.py diag_name=1 diag_name="['W7X_{nnnn}'.format(nnnn=nn) for nn in ['ECE_12','ECE_13','ECE_14','neL','TotECH','L53_LP01_U','L57_LP01_U']]" shot_list="[[20160302,s] for s in range(1,10)]"  overwrite_local=1 dev_name='W7X'  local_dir='/tmp' exceptions=Exception

New version
run pyfusion/examples/save_to_local.py "shot_list=shot_range([20160309,6],[20160309,9])" diag_name='["W7X_L53_LPALL","W7X_L57_LPALL","W7X_TotECH","W7X_neL","W7X_ECE_axis"]'
run pyfusion/examples/save_to_local.py "shot_list=shot_range([20160309,6],[20160309,9])" diag_name='["W7X_TotECH","W7X_L57_LPALL"]'

"""
from __future__ import print_function
from six.moves import input
import numpy as np
import os, sys, pickle, json
import time as tm

import pyfusion
from pyfusion.data.save_compress import discretise_signal as savez_new
import pyfusion.utils
from pyfusion.debug_ import debug_
from pyfusion.utils import process_cmd_line_args, pause_while
from pyfusion.utils.time_utils import utc_ns

if hasattr(pyfusion, 'NSAMPLES') and pyfusion.NSAMPLES != 0:
    if input('pyfusion.NSAMPLES is going to decimate - are you sure?').lower()[0]!='y':
        sys.exit(1)
    
pyfusion.reload_config()  # needed for W7M hacks - e.g if ROI is set specially

try:       # this allows usage on systems without all the new url features
    from pyfusion.data.shot_range import shot_range
    from pyfusion.acquisition.W7X.get_shot_info  import get_shot_utc
    from pyfusion.acquisition.W7X.find_shot_times import find_shot_times
    
except ImportError as reason:
    print('******* Warning - failed to import W7X stuff! ' + str(reason))
    shot_range = range
    
_var_defaults="""
dev_name='W7X'
readback=False
downsample=None
time_range=None
shot_list=[[20160310,9]]  # make it a list so that process_cmd_line_args is happy
compress_local=False
overwrite_local=True
save_kwargs = {} 
prefix=''  #'HeliotronJ_'
local_dir='/tmp'  # safe for linux and windows
exceptions = Exception  # a single exception or tuple of exceptions to be continued past [] or () will cause pyfusion to stop at the error
find_kws={}  # e.g. find_kws = "dict(diag=W7X_LTDU_LP20_I)"
save_in_RAW=1   # save in raw mode by default, so gain is not applied twice.
diag_name=["W7X_TotECH"] # Use a list - otherwise process_cmd will not accept a list
"""
exec(_var_defaults)
exec(process_cmd_line_args())

pyfusion.RAW=save_in_RAW  # FUDGE!!! until fixed

if local_dir is not '':
    if not os.path.isdir(local_dir):
        raise LookupError('local_dir {l} not found'.format(l=local_dir))
bads = []
goods = []


#s = pyfusion.get_shot(shot_number)
#s.load_diag(diag_name, savelocal=True, ignorelocal=(compress_local==None), downsample=True)

def getlocalfilename(shot_number, channel_name, local_dir=''):
    """
    At present, we assume the numpy savez method is used - other save options may be added later
    """
    if local_dir == '': # default to first path in localdatapath
        local_dir =  pyfusion.config.get('global', 'localdatapath').split('+')[0]
    # allow for multi-valued shot numbers - e.g. W7-X data with from,to utc
    if isinstance(shot_number, (tuple, list, np.ndarray)):
        fn = str(local_dir+'/{s0}_{s1}_{c}.npz'
                 .format(s0=shot_number[0], s1=shot_number[1], c=channel_name))
    else:
        fn = local_dir+'/{s}_{c}.npz'.format(s=shot_number, c=channel_name)
    return fn

# main

""" Logic:
time_range is always in seconds relative to t1 (or None)
shot can be a utc pair or a W7X shot number, or a W7<=M shot number
If asked to find_shot_times, return either found_utc or None, AND set this_time_range to [-1,0.3]
If we have utc shot number or a found_utc, this provides the fetch_utc - otherwise
use get_shot_utc to define shot_utc and set fetch_utc from shot_utc
If time_range is not None, adjust fetch_utc

For W7M, need to see if getdata accepts utcs, but in the meantime, fudge by changing the runtime ROI and pass a shot_number


"""
if len(np.shape(shot_list))==0:
    shot_list = [shot_list]
if len(shot_list) == 0:
    raise LookupError('No shots in list!')

if len(np.shape(diag_name)) == 0:
    diag_list = [diag_name]
else:
    diag_list = diag_name
    
for shot_number in shot_list:
    dev = pyfusion.getDevice(dev_name)
    if pause_while(os.path.join(local_dir, 'pause'), check=60) == 'quit':
        break
    else:
        pass

    pyfusion.RAW = save_in_RAW  # bdb kludge - fix and remove
    if find_kws != {}:
        found_utc = find_shot_times(shot=shot_number, **find_kws)
        print('Using threshold detection: {kws}'.format(kws=find_kws))
        if found_utc is None: #  meant to find, but failed
            # use the given time range, or if not set, a minimum amount enough to see problem
            this_time_range = time_range if time_range is not None else [-0.15,0.3]
        else:
            this_time_range = None  # we have a plasma range, we don't want to fiddle
    else:
        found_utc = None
        this_time_range = time_range
    if found_utc is not None:
        fetch_utc = found_utc
        utc0 = fetch_utc[0]
    elif shot_number[0] > 1e9:     # if it is a real shot
        fetch_utc = shot_number
        utc0 = fech_utc[0]
    else: # have to be careful - time_rnage None means get all - before and after t1
        shot_utc = get_shot_utc(shot_number)
        fetch_utc = shot_utc
        utc0 = shot_utc[0] + int(61 * 1e9)

    if this_time_range is not None:
        final_fetch_utc = [utc0 + int(tr * 1e9) for tr in this_time_range]
    else:
        final_fetch_utc = fetch_utc
        
    if dev_name == 'W7M':  # kludge set roi to control the time range
        print('dev.acq.roi = {dar}'.format(dar=dev.acq.roi),end=': ')
        if shot_number[0] < 990000:  # test shot - remember MDS shots are YYMMDD,NNN?
            mds_utc_offs = 0
        else:
            mds_utc_offs = get_shot_utc(shot_number)[0] + int(60*1e9)  # bdb kludgey - fix!!

        dev.acq.roi = ' '.join([str(tn) for tn in (final_fetch_utc - mds_utc_offs).tolist() + [100]])
        # use current ROI - crude!
        used_shot_number = shot_number
        utc_shot_number = shot_number  # ignore utcs for now
        # time has already been converted to roi and updated
        #  in pyfusion.config (line 135? I can't see it)
    else:
        used_shot_number = final_fetch_utc
        
    for diag in diag_list:
        this_save_kwargs = save_kwargs.copy()
        # for now, try to compress timebase for all shots - was just > 20180101
        if ('W7' in diag and 'LP' in diag) and shot_number[0]<20160000:
            # Try to selectively override delta time
            this_save_kwargs.update({"delta_encode_time": False})
            print('Set delta encode time to', this_save_kwargs['delta_encode_time'])
            
        diag = prefix + diag

        try:   # catch on a per diagnostic or multi bases
            params = pyfusion.conf.utils.get_config_as_dict('Diagnostic',diag)
            chan_list = []
            # now make a list of channels so the same code can be used by multi or single
            if (params['data_fetcher']
                == 'pyfusion.acquisition.base.MultiChannelFetcher'):
                for key in params:
                    if 'channel_' in key:
                        chan_list.append(params[key])
            else:
                chan_list = [diag]

            for diag_chan in chan_list:
                # this try except was added to catch errors on a diagnotic by
                # diagnostic basis.  Previously one bad killed all diags in the shot
                # There is some redundant code left over in outer try/ex loop
                try:
                    debug_(pyfusion.DEBUG, 1, key='before_getdata')
                    data = dev.acq.getdata(used_shot_number, diag_chan, no_cache=compress_local==False, exceptions=())
                    print(diag_chan, shot_number, end=' - ')

                    # the above will help stop saving over existing data.  This is important
                    # if we are replacing incorrect existing data..
                    # the next line also will protect if the cache file is new enough
                    if data is None:
                        raise LookupError('Data not found - maybe because use of cache is prevented')
                    if 'npz' in data.params['source'] and compress_local == False:
                        # this is to protect against accidental recompressions - 
                        raise Exception("Should not use local cached (npz) data unless you are compressing")

                    chan = data.channels
                    if downsample is not None:
                        data = data.downsample(downsample)

                    #no longer needed
                    #if time_range is not None and utc_shot_number is not None:
                    #    data = data.reduce_time(time_range, fftopt=True)

                    # I don't believe this test - always true!

                    if readback:
                        srb = pyfusion.get_shot(shot_number)
                        srb.load_diag(diag_chan, savelocal=False, ignorelocal=False)
                        srb==s

                    if (compress_local is not None):
                        from pyfusion.data.save_compress import discretise_signal as savez_new
                        from matplotlib.cbook import is_string_like

                        tb = data.timebase

                        if local_dir !='':
                            #probably should be chan.config_name here (not chan.name)
                            localfilename = getlocalfilename(
                                shot_number, chan.config_name, local_dir = local_dir)
                        else:
                            localfilename = getlocalfilename(shot_number, chan.config_name)

                        params = dict(name = diag_chan, device = dev_name, utc=data.utc, raw=save_in_RAW, host = pyfusion.utils.host())
                        if hasattr(data, 'params'):  # add the other params
                            params.update(data.params)
                        print('cal_info')
                        if hasattr(data, 'cal_info'):  # add the cal_date and comment
                            params.update(data.cal_info)

                        signal = data.signal

                        if os.path.isfile(localfilename) and not overwrite_local:
                            raise IOError('file {f} exists'.format(f=localfilename))

                        debug_(pyfusion.DEBUG,1, key='save_to_local')
                        savez_new(signal=signal, timebase=tb, filename=localfilename, 
                                  params = np.array(params),
                                  verbose=pyfusion.VERBOSE, **this_save_kwargs)
                        goods.append((shot_number,diag_chan))
                except IOError as reason:
                    if 'No space' in str(reason):
                        print('IO Error - no space')
                        sys.exit(1)
                except exceptions as chan_reason:
                    bads.append((shot_number, diag_chan, chan_reason.__repr__()))
            # redundant goods.append((shot_number,'whole thing'))
        except exceptions as reason:
            bads.append((shot_number,'whole thing',reason.__repr__()))
            print('skipping shot {s} because <<{r}>>'.format(r=reason, s=shot_number))
pfile = str('{s}_{dt}_save_local'
            .format(dt=tm.strftime('%Y%m%d%H%M%S'), 
                    s=str(shot_number).replace('(','').replace(')','')
                    .replace(',','_').replace(' ','')))

print('See bads for {l} errors, also goods ({g}), and in {pfile}'.format(l=len(bads), g=len(goods), pfile=pfile))
if pyfusion.RAW == 0: print('remember to reset pyfusion.RAW')
try:
    json.dump(dict(bads=bads, goods=goods), open(pfile+'.json','w'))
except:
    pickle.dump([bads,goods], open(pfile+'.pickle','w'))
