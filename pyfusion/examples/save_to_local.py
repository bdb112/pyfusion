""" 
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

# example with a two component shot number
run pyfusion/examples/save_to_local.py shot_list='[[20160225,s] for s in range(1,50)]'  overwrite_local=1 dev_name='W7X' diag_name='W7X_L57_LP01_08'  local_dir='/data/datamining/local_data/' exception=Exception

from the old pyfusion - may need to tidy up
_PYFUSION_TEST_@@local_dir=/tmp/ overwrite_local=True

# this example works Mar 8 2016 - need shot list and diag list to avoid problems
run  pyfusion/examples/save_to_local.py diag_name=1 diag_name="['W7X_L53_LP{nnnn}_I'.format(nnnn=nn) for nn in [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22]]" shot_list="[[20160302,s] for s in range(1,10)]"  overwrite_local=1 dev_name='W7X'  local_dir='/tmp' exception=Exception

# the rest of the stuff
run  pyfusion/examples/save_to_local.py diag_name=1 diag_name="['W7X_{nnnn}'.format(nnnn=nn) for nn in ['ECE_12','ECE_13','ECE_14','neL','TotECH','L53_LP01_U','L57_LP01_U']]" shot_list="[[20160302,s] for s in range(1,10)]"  overwrite_local=1 dev_name='W7X'  local_dir='/tmp' exception=Exception

New version
run pyfusion/examples/save_to_local.py "shot_list=shot_range([20160309,6],[20160309,9])" diag_name='["W7X_L53_LPALL","W7X_L57_LPALL","W7X_TotECH","W7X_neL","W7X_ECE_axis"]'
run pyfusion/examples/save_to_local.py "shot_list=shot_range([20160309,6],[20160309,9])" diag_name='["W7X_TotECH","W7X_L57_LPALL"]'

"""
import pyfusion
from pyfusion.data.save_compress import discretise_signal as savez_new
import pyfusion.utils
import numpy as np
import os
from pyfusion.debug_ import debug_

from pyfusion.utils import process_cmd_line_args

from pyfusion.data.shot_range import shot_range

_var_defaults="""
diag_name = 'SLOW2k'  # Use a list - otherwise process_cmd will not accept a list
dev_name='W7X'
readback=False
downsample=None
time_range=None
shot_list=[27233]  # make it a list so that process_cmd_line_args is happy
compress_local=1
overwrite_local=True
save_kwargs = {} 
prefix=''  #'HeliotronJ_'
local_dir='/tmp'  # safe for linux and windows
exception = Exception
pyfusion.RAW=1   # save in raw mode by default, so gain is not applied twice.
diag_name=["W7X_TotECH"]
"""
exec(_var_defaults)
exec(process_cmd_line_args())

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
if len(np.shape(shot_list))==0:
    shot_list = [shot_list]

if len(np.shape(diag_name)) == 0:
    diag_list = [diag_name]
else:
    diag_list = diag_name
    
for shot_number in shot_list:
    dev = pyfusion.getDevice(dev_name)
    for diag in diag_list:
        if 'W7' in diag and 'LP' in diag:
            print('override encode time')
            save_kwargs={"delta_encode_time":False}

        diag = prefix + diag

        try:   # catch on a per diagnostic or multi bases
            params = pyfusion.conf.utils.get_config_as_dict('Diagnostic',diag)
            chan_list = []
            if (params['data_fetcher']
                == 'pyfusion.acquisition.base.MultiChannelFetcher'):
                for key in params:
                    if 'channel_' in key:
                        chan_list.append(params[key])
            else:
                chan_list = [diag]

            for diag_chan in chan_list:
                data = dev.acq.getdata(shot_number, diag_chan)
                chan = data.channels
                if downsample is not None:
                    data = data.downsample(downsample)

                if time_range is not None:  # not tested!!
                    data = data.reduce_time(time_range, fftopt=True)

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

                    params = dict(name = diag_chan, device = dev_name, utc=data.utc, raw=pyfusion.RAW, host = pyfusion.utils.host())
                    if hasattr(data, 'params'):  # add the other params
                        params.update(data.params)

                    signal = data.signal

                    if os.path.isfile(localfilename) and not overwrite_local:
                        raise IOError('file {f} exists'.format(f=localfilename))

                    debug_(pyfusion.DEBUG,1, key='save_to_local')
                    savez_new(signal=signal, timebase=tb, filename=localfilename, 
                              params = np.array(params),
                              verbose=pyfusion.VERBOSE, **save_kwargs)
            goods.append((shot_number,'whole thing'))
        except exception as reason:
            bads.append((shot_number,'whole thing'))
            print('skipping shot {s} because {r}'.format(r=reason, s=shot_number))
print('See bads for {l} errors (also goods)'.format(l=len(bads)))
