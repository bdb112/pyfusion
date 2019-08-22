""" simple example to plot a spectrogram, uses command line arguments

    run pyfusion/examples/plot_specgram shot_number=69270

    See process_cmd_line_args.py
    channel_number
    shot_number
    diag_name
_PYFUSION_TEST_@@NotSkip
"""

from __future__ import division
import pyfusion
import pylab as pl
import matplotlib.cm as cm  # import for convenience of user input.

def set_clim(clim, use_list=None):
    if use_list is None:
        use_list = ax_list
    print('Setting clim for last image - for Nth from bottom do\n'
          'ax_list[N].get_images()[0].set_clim(clim)')
    use_list[0].get_images()[0].set_clim(clim)
    pl.show()
    
_var_defaults="""
dev_name='H1Local'   # 'LHD'
dev_name='LHD'
# ideally should be a direct call, passing the local dictionary

shot_number = None
diag_name = ""
NFFT=256
noverlap=None
time_range = None
channel_number=0
hold=0
clim=None  #  if 'show' show the clim in the legend - (note - "clim='show'")
xlim=None
ylim=None
cmap=cm.jet   # see also cm.gray_r etc
stop=False
hspace = None
"""

exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

device = pyfusion.getDevice(dev_name)

if dev_name == 'LHD':
    if shot_number is None: shot_number = 27233
    if diag_name is "": diag_name= 'MP'
elif dev_name[0:1] == "H1":
    if shot_number is None: shot_number = 69270
    if diag_name is "": diag_name = "H1DTacqAxial"
elif dev_name == "HeliotronJ":
    if shot_number is None: shot_number = 27633
    if diag_name is "": diag_name = "HeliotronJ_MP2"

exec(pyfusion.utils.process_cmd_line_args())

if noverlap is None: noverlap = NFFT//2

## time_range = process_time_range(time_range)
# Shortcut for small range: (should work for 3 element also (3rd is interval))
# so that time_range=[3,.1] --> [2.9, 3.1] and [1,1] -> [0,2]
if (time_range is not None and
    (time_range[0] != 0) and
    (abs(time_range[1]/time_range[0]) <= 1) and
    (time_range[1] < time_range[0])):
    print('Using delta time range in multi channel fetcher ')
    dt = time_range[1]
    time_range[1]  = time_range[0] + dt
    time_range[0] -= dt

d = device.acq.getdata(shot_number, diag_name, time_range=time_range, contin=not stop)
# not needed now that getdata respects time range
#if time_range != None:
#    dr = d.reduce_time(time_range)
#else:
#    dr = d
dr = d.subtract_mean()
ax_list = dr.plot_spectrogram(noverlap=noverlap, NFFT=NFFT, channel_number=channel_number, hold=hold, cmap=cmap, clim=clim, xlim=xlim, ylim=ylim, hspace=hspace)
pyfusion.utils.warn('set_clim([-80, -10] to change clims')
