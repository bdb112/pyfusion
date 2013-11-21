""" Saves a set of channels (diagnostic) diag_name in local files
compress_local does an "in-place" compress if True, and if a string,
puts the result in the local directory compress_local. The compress_local option
doesn't check if the file is already compressed!

Examples:

# This compresses local data using newer compression methods
run pyfusion/examples/save_to_local.py shot_number=27233 compress_local=1 diag_name="MP"
# note the extra quotes on compress_local below, as its default is None
run examples/save_to_local.py shot_number=18993 'compress_local="/tmp"' diag_name="MP2010"

(or in windows  'compress_local="c:/cygwin/tmp"')
# tuning compression parameters - (this example shows very little difference
run examples/save_to_local.py shot_number=18993 'compress_local="c:/cygwin/tmp"' diag_name="mirnov_small" save_kwargs='{"delta_encode_signal":True}'

    You can cut down file size to include only data between
    PYFUSION_SHOT_T_MIN and MAX by saving to local (twice).
    The first time, data are saved in full, but only the desired part is
    retrieved form local.  You can save a second time, to a different path
    and in that case, only the MIN to MAX are saved locally.

"""
import pyfusion
from pyfusion.data.save_compress import discretise_signal as savez_new
import pyfusion.utils

_var_default="""
diag_name = 'SLOW2k'
diag_name = "MP1"
dev_name='LHD'
readback=False
shot_number=33343
compress_local=None
save_kwargs = {} 
"""
exec(_var_default)
exec(pyfusion.utils.process_cmd_line_args())


#s = pyfusion.get_shot(shot_number)
#s.load_diag(diag_name, savelocal=True, ignorelocal=(compress_local==None), downsample=True)

h1=pyfusion.getDevice(dev_name)
data=h1.acq.getdata(shot_number,diag_name)

def getlocalfilename(shot_number, channel_name, local_dir=None):
	"""
	At present, we assume the numpy savez method is used - other save options may be added later
	"""
	if local_dir == None: # default to first path in localdatapath
		local_dir =  pyfusion.config.get('global', 'localdatapath').split(':')[0]
	return local_dir+'/%d_%s.npz' %(shot_number, channel_name)


# That's all for simple save

# I don't believe this test - always true!

if readback:
    srb = pyfusion.get_shot(shot_number)
    srb.load_diag(diag_name, savelocal=False, ignorelocal=False)
    srb==s

if (compress_local is not None):
    from pyfusion.data.save_compress import discretise_signal as savez_new
    from matplotlib.cbook import is_string_like

    tb = data.timebase
    for (c,chan) in enumerate(data.channels):
        if is_string_like(compress_local):
            localfilename = getlocalfilename(
                shot_number, chan.name, local_dir = compress_local)
        else:
            localfilename = getlocalfilename(shot_number, chan.name)

        signal = data.signal[c]
        savez_new(signal=signal, timebase=tb, filename=localfilename,
                  verbose=pyfusion.VERBOSE, **save_kwargs)
