""" This is very old, has been updated just a little - gets an error grabbing k_h??
test generation of flucstrucs.
from gen_fs_local, try to generalise
  exception=None catches no exceptions, so you get a stop and a traceback
  exception=Exception catches all, so it continues, but no traceback
python dm/gen_fs.py shot_range=[27233] exception=None
_PYFUSION_TEST_@@n_samples=8192

"""
import subprocess, sys, warnings
from numpy import sqrt, mean
import pyfusion as pf
import traceback

_var_defaults="""
device_name = 'H1Local' # 'LHD'
shot_range = [76616]
n_samples = 512
overlap=1.0

exception=Exception
time_range = None
min_svs=1
max_H=0.97
"""

exec(_var_defaults)

import pyfusion.utils
exec(pf.utils.process_cmd_line_args())
dev = pf.getDevice(device_name)

#min_shot = 84000
#max_shot = 94000
#every_nth = 10

#shot_range = range(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

if device_name == 'LHD':
    shot_range = [27233]
    diag_name= 'MP'
  #shot_range = range(90090, 90110)
elif device_name.find('H1')>=0:
    shot_range = [76616]
    diag_name = 'H1ToroidalAxial' # 'H1TestPoloidal' #'H1Poloidal1'  #letter l number 1 

# ids are usually handled by sqlalchemy, without SQL we need to look after them ourselves
fs_id = 0

# ideally should be a direct call, passing the local dictionary
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

for shot in shot_range:
    try:
        d = dev.acq.getdata(shot, diag_name)
        if time_range != None:
            d.reduce_time(time_range)
        sections = d.segment(n_samples, overlap)
        print(d.history, len(sections))

        for ii,t_seg in enumerate(sections):
            fs_set = t_seg.flucstruc()
            for fs in fs_set:
                if fs.H < max_H and fs.p>0.01 and len(fs.svs())>min_svs:
                    if pf.USE_ORM:  # strange way to access if in ORM mode
                        phases = ' '.join(["%5.2f" % fs.dphase[dpi].delta for dpi in fs.dphase.data_items])
                    else:
                        # this line is either wrong or very old
                        #phases = ' '.join(["@%5.2f" % dpo.item.delta for dpo in fs.dphase])
                        phases = ' '.join(["%5.2f" % dpo.delta for dpo in fs.dphase])
                    RMS_scale = sqrt(mean(t_seg.scales**2))
                    adjE = fs.p*fs.E*RMS_scale**2
                    print ("%d %7.4g %s %6.3g %6.3f %.2f %.3f %.3f    %s" % (
                            shot, fs.t0, "{0:8b}".format(fs._binary_svs), 
                            fs.freq/1000., sqrt(fs.E*fs.p)*RMS_scale, fs.a12, fs.p, fs.H, phases))
    
        # the -f stops the rm cmd going to the terminal
#        subprocess.call(['/bin/rm -f /data/tmpboyd/pyfusion/FMD*%d*' %shot], shell=True)
#        subprocess.call(['/bin/rm -f /data/tmpboyd/pyfusion/SX8*%d*' %shot], shell=True)
    except exception:
#ZeroDivisionError is a typical exception that won't happpen to allow
# traceback to show - however () is better - None will work only in python2
        traceback.print_exc()

        warnings.warn(str('shot %d not processed: %s' % 
                          (shot,sys.exc_info()[1].__repr__())))
