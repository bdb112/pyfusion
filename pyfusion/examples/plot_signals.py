""" see plot_signal_trivial for bare bones 
plots a single or multi-channel signal
  hold=0 use a new figure
       1 Reuse figure
       2 make a second y axis on the same figure
This example shows bad motorboating
run pyfusion/examples/plot_signals dev_name='H1Local' diag_name='H1Poloidal1' shot_number=76887
# multichannel example
run pyfusion/examples/plot_signals dev_name='H1Local' diag_name='ElectronDensity' shot_number=76887
run pyfusion/examples/plot_signals diag_name='LHD_n_e_array' shot_number=42137 sharey=1
# npz local files
run pyfusion/examples/plot_signals.py dev_name='HeliotronJ' shot_number=50136 diag_name=HeliotronJ_MP_array
# Okada style files
run pyfusion/examples/plot_signals.py dev_name='HeliotronJ' shot_number=58000 diag_name=HeliotronJ_MP_array
"""
import pyfusion
from pyfusion.data.plots import myiden, myiden2, mydiff
from pyfusion.utils.time_utils import utc_ns
from pyfusion.utils import wait_for_confirmation
import matplotlib.pyplot as plt

_var_defaults = """
dev_name = "W7X"
diag_name = "W7X_TotECH"
shot_number = [20160308,39]
# dev_name = "LHD"; diag_name = "MP1" ; shot_number = 27233
sharey=1  #  1 uses same y axis for all, 2 for all but the top one (3 for all but the top two etc).
decimate=1   #  1 is no decimation, 10 is by 10x
fun=myiden
fun2=myiden2
plotkws={}
hold=0  # 0 means erase, 1 means hold, 2 means twinx
labeleg='False'
t_range=None
time_range=None
t0=0
stop=True  # if False, push on with missing data  - only makes sense for multi channel, and may only work when pyfusion.DEBUG is a number.
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args, choose_one
exec(process_cmd_line_args())

time_range = choose_one(time_range, t_range)
# save the start utc if there is already a 'data' object (***need to run -i to keep)
if 'W7X' in dev_name:
    if True:  # this catches the use of run -i, and data being there from a previous plot
        if 'data' in locals() and hold !=0:
            if 'allow_utc' not in locals():
                resp = wait_for_confirmation('Allow the utc0 fudge? (n/q/y)', exit_for_n=False)
                allow_utc = resp == 'y'
            if 'allow_utc' in locals() and allow_utc and utc0 not in locals():
                utc0 = data.utc[0] 

## move this to a function: time_range = process_time_range(time_range)
# Shortcut for small range: (should work for 3 element also (3rd is interval))
# so that time_range=[3,.1] --> [2.9, 3.1] and [1,1] -> [0,2]
if (time_range is not None and
    (time_range[0] != 0) and
    (abs(time_range[1]/time_range[0]) <= 1) and
    (time_range[1] < time_range[0])):
    print('Using delta time range ')
    dt = time_range[1]
    time_range[1]  = time_range[0] + dt
    time_range[0] -= dt

dev = pyfusion.getDevice(dev_name)
data = dev.acq.getdata(shot_number,diag_name, contin=not stop, time_range=time_range)
if data is None:
    raise LookupError('data not found for {d} on shot {sh}'
                      .format(d=diag_name, sh=shot_number))
#  this is only here if getdata fails to select time range not needed now
#if time_range is not None:
#    data = data.reduce_time(time_range)

if hold==0: plt.figure()
elif hold==2:
    if 'first_ax' in locals() and first_ax is not None:
        plt.sca(ax)
    else:
        first_ax = plt.gca()
        ax = first_ax.twinx()
        plt.sca(ax)
        labeleg='True'

# should no longer be needed - comment out for now until we test more of the old routines.
if 'W7X' in dev_name:
    if t0 is None:  # if t0 NOT given (i.e. None) then  we are asking to fudge it
        pgms = pyfusion.acquisition.W7X.get_shot_list.get_programs(shot=shot_number)
        pgm = pgms['{d}.{s:03d}'.format(d=shot_number[0], s=shot_number[1])]
        utc0 = pgm['trigger']['5'][0]
        # maybe should be in data/plots.py, but config_name not fully implemented
    if 'allow_utc' in locals() and allow_utc and 'utc0' in locals() and utc0 is not None:  # if t0 is given, and there an utc- and allow_utc is True
        pyfusion.utils.warn("using previous utc - can give trouble for data after 2016 - don't use run -i")
        print("\n***** using previous utc - can give trouble for data after 2016 - set allow_utc=False or don't use run -i ")
        t0 = (utc0 - data.utc[0])/1e9
        
data.plot_signals(suptitle='shot {shot}: '+diag_name, t0=t0, sharey=sharey,
                  downsamplefactor=max(1, decimate),
                  fun=fun, fun2=fun2, labeleg=labeleg, **plotkws)

# for shot in range(76620,76870,10): dev.acq.getdata(shot,diag_name).plot_signals()
if labeleg:
    plt.legend(loc='best')
