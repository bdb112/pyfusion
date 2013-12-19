""" Try band limited generation of flucstrucs
from gen_fs_local Feb 2013
#exception=None catches no exceptions, so you get a stop and a traceback
#exception=Exception catches all, so it continues, but no traceback
python dm/gen_fs.py shot_range=[27233] exception=None

71 secs to sqlite.txt (9MB) (21.2k in basedata, 3.5k fs, 17.7k float_delta
34 secs with db, but no fs_set.save()
17 secs to text 86kB - but only 1k fs

Modify to output to a file or stdout - not perfect yet
Advantage of stdout is that everything is recorded and each new run gets a new process - less chanceof memory exhaustion etc
Advantage of file output is that it is easier to run simple cases and test cases.

# HeliotronJ example: Using the "long form" of the probes - needed if in the same file
# as LHD with the same name.
run pyfusion/examples/gen_fs_bands.py dev_name='HeliotronJ' diag_name='HeliotronJ_MP_array' time_range=[100,250] shot_range=[50000] seg_dt=1 info=1

"""
import subprocess, sys, warnings
from pyfusion.utils.utils import warn
from numpy import sqrt, mean, argsort, average, random
import numpy as np
import pyfusion
from pyfusion.debug_ import debug_
import sys
from time import sleep, time as seconds
import os
from pyfusion.data.utils import find_signal_spectral_peaks, subdivide_interval
from pyfusion.data.pyfusion_sigproc import find_shot_times

# implement some extras from gen_fs_local
def timeinfo(message, outstream=sys.stdout):
    if show_times == 0: return
    global st, st_0
    try:
        dt = seconds()-st
    except:
        outstream.writelines('Time:' + message + '(first call)\n')
        st = seconds()
        st_0 = seconds()
        return
    outstream.writelines("Time: {m} in {dt:.2f}/{t:.2f}\n".format(m=message, dt=dt, t=seconds()-st_0))
    st = seconds()
    return

_var_default="""
dev_name = 'LHD'

#min_shot = 84000
#max_shot = 94000
#every_nth = 10

#shot_range = range(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

debug=0
show_times=1
shot_range = [27233]
#shot_range = range(90090, 90110)
n_samples = None  # was 512
overlap=1.0
diag_name= 'MP2010'
exception=Exception
time_range = None
min_svs=2
max_H=0.97
info=2
separate=1
method='rms'
seg_dt=1.5e-6 # time interval - overrides n_samples if None
df = 2e3  #Hz
fmax = None
max_bands = 4
toff=0                 # offset time needed for VSL retrieve - try retrieve16?
outfile="None"
# ids are usually handled by sqlalchemy, without SQL we need to look after them ourselves
fs_id = 0
"""
exec(_var_default)

# ideally should be a direct call, passing the local dictionary
import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())
from pyfusion.data.filters import next_nice_number

# Simple redirect of output:  this is not the perfect way - maybe look closer
#  web says we should consider context manager and thread safety
if outfile is None or outfile == 'None':  # stdout
    write = sys.stdout.write
    close = sys.stdout.flush
else:
    fd = open(outfile, 'w+')
    write = fd.write
    close = fd.close

count = 0  #  we print the header right before the first data
dev = pyfusion.getDevice(dev_name)
for shot in shot_range:
    while(os.path.exists(pyfusion.root_dir+'/pause')):
        print('paused until '+ pyfusion.root_dir+'/pause'+ ' is removed')
        sleep(int(20*(1+random.uniform())))  # wait 20-40 secs, so they don't all start together

    timeinfo('start shot {s}'.format(s=shot))
    try:
        d = dev.acq.getdata(shot, diag_name)
        timeinfo('data read')
        n_channels = len(d.channels)
        dt = np.average(np.diff(d.timebase))
        if n_samples is None:
            n_samples = next_nice_number(seg_dt/dt)
            write('Choosing {N} samples to cover {seg_dt}\n'
                  .format(N=n_samples,  seg_dt=seg_dt))
            
        if time_range != None:
            if type(time_range)==str: # strings are recipes for finding shot times
                time_range = find_shot_times(dev, shot, time_range)
            d.reduce_time(time_range, copy=False)

        sections = d.segment(n_samples, overlap)
        write('<<{h}, {l} sections, version = {vsn}\n'
              .format(h=d.history, l=len(sections), 
                      vsn=pyfusion.version.get_version('verbose')))
        try:
            for k in pyfusion.conf.history.keys():
                # the || prevent lines starting with a number
                write('||'+pyfusion.conf.history[k][0].split('"')[1])
                if info>1: sys.stdout.writelines(pyfusion.conf.utils.dump())
        except: pass    

        ord_segs = []
        for ii,t_seg in enumerate(sections):
            ord_segs.append(t_seg)
        ord = argsort([average(t_seg.timebase) for t_seg in ord_segs])
        if fmax == None:
            fmax = 0.5/np.average(np.diff(ord_segs[0].timebase)) - df

        timeinfo('beginning flucstruc loop')

        for idx in ord:
            t_seg = ord_segs[idx]
            if max_bands>1:              # keep it simple if one band
                (ipk, fpk, apk) = find_signal_spectral_peaks(t_seg.timebase, t_seg.signal[0],minratio=0.1)
                w_in_range = np.where(fpk < fmax)[0]
                (ipk, fpk, apk) = (ipk[w_in_range], fpk[w_in_range], apk[w_in_range])
                if len(ipk)>max_bands: 
                    warn('too many peaks - reducing to {mb}'.format(mb=max_bands))
                    fpk = np.sort(fpk[np.argsort(apk)[-(max_bands+1):]])

                
                (lfs, hfs) = subdivide_interval(np.append(fpk, fmax), debug=0, overlap=(df/2,df*2))
            else:
                (lfs,hfs) = ([0],[fmax])

            for i in range(len(lfs)):
                (frlow,frhigh)= (lfs[i],hfs[i])
                f_seg = t_seg.filter_fourier_bandpass(
                    [lfs[i],hfs[i]], [lfs[i]-df,hfs[i]+df]) 
                fs_set = f_seg.flucstruc(method=method, separate=separate)
                for fs in fs_set:
                    if count==0: 
                        # show history if info says to, and avoid starting line with a digit
                        if info > 0: write('< '+fs.history.
                                           replace('\n201','\n< 201'))
                        # Version number should end in digit!
                        SVtitle_spc = (n_channels - 2)*' '
                        write('\nShot    time   {spc}SVS    freq  Amp    a12   p    H     frlow frhigh  cpkf  fpkf  {np:2d} Phases       Version 0.7 \n'
                              .format(np=len(fs.dphase),spc=SVtitle_spc))
                    count += 1
                    if fs.H < max_H and fs.p>0.01 and len(fs.svs())>=min_svs:
                        phases = ' '.join(["%5.2f" % j.delta for j in fs.dphase])
                        # was t_seg.scales, but now it is copies, t_seg is not updated 
                        RMS_scale = sqrt(mean(fs.scales**2)) 
                        adjE = fs.p*fs.E*RMS_scale**2
                        debug_(debug,2)
                        if pyfusion.orm_manager.IS_ACTIVE: 
                        # 2-2.05, MP: 11 secs, 87  , commit=False 10.6 secs 87
                        # commits don't seem to reduce time.    
                            fs.save(commit=True)           
                        else: 
                            # 20121206 - time as %8.5g (was 7.4) 
                            # caused apparent duplicate times
                            # Aug 12 2013 - changed amp to sqrt(f)*RMS scale - like plot_svd.
                            SV_fmt = "{{0:{w}b}}".format(w=2+n_channels)
                            write ("%d %8.5g %s %6.3g %6.5f %.2f %.3f %.3f %5.1f %5.1f %5.1f %5.1f %s\n" % (
                                    #shot, fs.t0, "{0:11b}".format(fs._binary_svs), 
                                    shot, fs.t0-toff, SV_fmt.format(fs._binary_svs), 
                                    fs.freq/1000., sqrt(fs.p)*RMS_scale, # was sqrt(fs.E*fs.p)*RMS_scale see above
                                    fs.a12, fs.p, fs.H, frlow/1e3, frhigh/1e3, 
                                    fs.cpkf, fs.fpkf, #fs.E, fs.E is constant!
                                    phases))

        # the -f stops the rm cmd going to the terminal
#        subprocess.call(['/bin/rm -f /data/tmpboyd/pyfusion/FMD*%d*' %shot], shell=True)
#        subprocess.call(['/bin/rm -f /data/tmpboyd/pyfusion/SX8*%d*' %shot], shell=True)
    except exception:
#set Exception=None to allow traceback to show - it can never happen
# otherwise, the warnigns.warn will prevent multiple identical messages
        warning_msg = str('shot %d not processed: %s' % 
                          (shot,sys.exc_info()[1].__repr__()))
        warnings.warn(warning_msg,stacklevel=2)

if pyfusion.orm_manager.IS_ACTIVE: 

    pyfusion.orm_manager.Session().commit()
    pyfusion.orm_manager.Session().close_all()

timeinfo('\n######## {c} fs finished'.format(c=count), sys.stderr) # stderr so that read_text is not fooled.
# should really put a sentinel here.
close()
