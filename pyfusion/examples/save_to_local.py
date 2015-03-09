""" Saves a set of channels (diagnostic) diag_name in local files
compress_local does an "in-place" compress if True, and if a string,
puts the result in the local directory compress_local. The compress_local option
doesn't check if the file is already compressed!

Examples:

# This compresses local data using newer compression methods
run pyfusion/examples/save_to_local.py shot_list=27233 compress_local=1 diag_name="MP"
# note the extra quotes on compress_local below, as its default is None
run examples/save_to_local.py shot_list=18993 'compress_local="/tmp"' diag_name="MP2010"

(or in windows  'compress_local="c:/cygwin/tmp"')
# tuning compression parameters - (this example shows very little difference
run examples/save_to_local.py shot_number=18993 'compress_local="c:/cygwin/tmp"' diag_name="mirnov_small" save_kwargs='{"delta_encode_signal":True}'

from the old pyfusion - may need to tidy up

"""
import pyfusion
from pyfusion.data.save_compress import discretise_signal as savez_new
import pyfusion.utils
import numpy as np

_var_default="""
diag_name = 'SLOW2k'
diag_name = "MP1"
dev_name='LHD'
readback=False
shot_list=33343  # number or a list
compress_local=None
save_kwargs = {} 
prefix=''  #'HeliotronJ_'
"""
exec(_var_default)
exec(pyfusion.utils.process_cmd_line_args())


#s = pyfusion.get_shot(shot_number)
#s.load_diag(diag_name, savelocal=True, ignorelocal=(compress_local==None), downsample=True)

def getlocalfilename(shot_number, channel_name, local_dir=None):
    """
    At present, we assume the numpy savez method is used - other save options may be added later
    """
    if local_dir is None: # default to first path in localdatapath
        local_dir =  pyfusion.config.get('global', 'localdatapath').split(':')[0]
    return local_dir+'/%d_%s.npz' %(shot_number, channel_name)

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
        h1=pyfusion.getDevice(dev_name)
        data=h1.acq.getdata(shot_number,diag)

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
                if is_string_like(compress_local):
                    #probably should be chan.config_name here (not chan.name)
                    localfilename = getlocalfilename(
                        shot_number, chan.config_name, local_dir = compress_local)
                else:
                    localfilename = getlocalfilename(shot_number, chan.config_name)

                params = dict(name = diag, device = dev_name)
                if hasattr(data, 'params'):  # add the other params
                    params.update(data.params)

                if singleton:
                    signal = data.signal
                else: 
                    signal = data.signal[c]

                savez_new(signal=signal, timebase=tb, filename=localfilename, 
                          params = np.array(params),
                          verbose=pyfusion.VERBOSE, **save_kwargs)
