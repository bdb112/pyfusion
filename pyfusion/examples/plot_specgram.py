""" simple example to plot a spectrogram, uses command line arguments

    run pyfusion/examples/plot_specgram shot_number=69270

    See process_cmd_line_args.py
    channel_number
    shot_number
    diag_name
_PYFUSION_TEST_@@NotSkip
"""

from __future__ import division
import pyfusion as pf
import pylab as pl
import matplotlib.cm as cm  # import for convenience of user input.

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

device = pf.getDevice(dev_name)

if dev_name == 'LHD':
    if shot_number is None: shot_number = 27233
    if diag_name is "": diag_name= 'MP'
elif dev_name[0:1] == "H1":
    if shot_number is None: shot_number = 69270
    if diag_name is "": diag_name = "H1DTacqAxial"
elif dev_name == "HeliotronJ":
    if shot_number is None: shot_number = 27633
    if diag_name is "": diag_name = "HeliotronJ_MP2"

exec(pf.utils.process_cmd_line_args())

if noverlap is None: noverlap = NFFT//2

d = device.acq.getdata(shot_number, diag_name, time_range=time_range, contin=not stop)
if time_range != None:
    dr = d.reduce_time(time_range)
else:
    dr = d
dr = dr.subtract_mean()
ax_list = dr.plot_spectrogram(noverlap=noverlap, NFFT=NFFT, channel_number=channel_number, hold=hold, cmap=cmap, clim=clim, xlim=xlim, ylim=ylim, hspace=hspace)
