""" Saves a channel or set of channels (diagnostic) diag_name in local files
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
run pyfusion/examples/save_to_local.py shot_list='[[20160225,s] for s in range(1,50)]'  overwrite_local=1 dev_name='W7X' diag_name='W7X_L57_LP1_8'  local_dir='/data/datamining/local_data/' exception=Exception

from the old pyfusion - may need to tidy up
_PYFUSION_TEST_@@local_dir=/tmp/ overwrite_local=True
"""
import pyfusion
from pyfusion.data.save_compress import discretise_signal as savez_new
import pyfusion.utils
import numpy as np
import os

from pyfusion.utils import process_cmd_line_args

_var_defaults="""
diag_name = 'SLOW2k'
diag_name = "MP1"
dev_name='LHD'
readback=False
downsample=None
time_range=None
shot_list=33343  # number or a list
shot_list=[27233]
compress_local=1
overwrite_local=False
save_kwargs = {} 
prefix=''  #'HeliotronJ_'
local_dir=''
exception= Exception
pyfusion.RAW=1   # save in raw mode by default, so gain is not applied twice.
"""
exec(_var_defaults)
exec(process_cmd_line_args())

if 'W7' in diag_name and 'LP' in diag_name:
    print('override encode time')
    save_kwargs={"delta_encode_time":False}

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
    
for diag in diag_list:
    diag = prefix+diag
    for shot_number in shot_list:
        dev = pyfusion.getDevice(dev_name)
        try: 
            data = dev.acq.getdata(shot_number, diag)
            if downsample is not None:
                data = data.downsample(downsample)

            if time_range is not None:  # not tested!!
                data = data.reduce_time(time_range, fftopt=True)

            # I don't believe this test - always true!

            if readback:
                srb = pyfusion.get_shot(shot_number)
                srb.load_diag(diag, savelocal=False, ignorelocal=False)
                srb==s

            if (compress_local is not None):
                from pyfusion.data.save_compress import discretise_signal as savez_new
                from matplotlib.cbook import is_string_like

                tb = data.timebase
                singleton =  len(np.shape(data.channels))==0
                if singleton:  chan_list = [data.channels]
                else: chan_list = data.channels

                for (c,chan) in enumerate(chan_list):
                    if local_dir !='':
                        #probably should be chan.config_name here (not chan.name)
                        localfilename = getlocalfilename(
                            shot_number, chan.config_name, local_dir = local_dir)
                    else:
                        localfilename = getlocalfilename(shot_number, chan.config_name)

                    params = dict(name = diag, device = dev_name, utc=data.utc, raw=pyfusion.RAW)
                    if hasattr(data, 'params'):  # add the other params
                        params.update(data.params)

                    try:
                        if singleton:
                            signal = data.signal
                        else: 
                            signal = data.signal[c]
                    except exception as reason:
                        print('failed to read shot {shot_number}, {r} {args}'
                              .format(shot=shot, r=reason, args=r.args))

                    if os.path.isfile(localfilename) and not overwrite_local:
                        raise IOError('file {f} exists'.format(f=localfilename))

                    savez_new(signal=signal, timebase=tb, filename=localfilename, 
                              params = np.array(params),
                              verbose=pyfusion.VERBOSE, **save_kwargs)

        except exception as reason:
            print('skipping shot {s} because {r}'.format(r=reason, s=shot_number))
