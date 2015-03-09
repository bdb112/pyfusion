""" Plot the svd of a diag, either one with arbitrary time bounds
or the sequence of svds of numpts starting at start_time (sec)
keys typed at the terminal allow stepping backward and forward.

Works from RAW data.

Typical usage : run  dm/plot_svd.py start_time=0.01 "normalise='v'" use_getch=0
      separate [=1] if True, normalisation is separate for each channel
Dave's checkbuttons on plot_svd don't work unless you use hold.
running outside pyfusion is more reliable

Note that the code is hard wired to reset hold after a "hold" so that you can continue

Note: Unless deepcopy is used in data/base.py,  data is changed after processing, so it will be a little different when revisited.
Better to re-run WITHOUT -i to check 
"""

def order_fs(fs_set, by='p'):
    """ Dave's code returns an unordered set - need to order by singular value (desc)
    """
    fsarr_unsort=[]
    for fs in fs_set: fsarr_unsort.append(fs)
    if by == 'p': revord = argsort([fs.p for fs in fsarr_unsort ])
    else: raise ValueError, str(" sort order %s not supported " % by)
    fs_arr = []
    for ind in revord: fs_arr.append(fsarr_unsort[ind])
    fs_arr.reverse() # in place!
    return(fs_arr)


import subprocess, sys, warnings
from numpy import sqrt, argsort, average, mean, pi
import pyfusion as pf
import pyfusion.utils
import pylab as pl
import numpy as np
from copy import deepcopy


try:
    import getch
    use_getch = True
except:
    use_getch = False
print(" getch is %savailable" % (['not ', ''][use_getch]))

plot_mag = 0
plot_phase = 0

diag_name = ''
dev_name='H1Local'   # 'LHD'
hold=0
exception=Exception
time_range = None
channel_number=0
lowpass = None  # set to corner freq for lowpass filter
highpass = None  # set to corner freq for lowpass filter
start_time = None
numpts = 512
normalise='0'
myfilter1p5=dict(passband=[1e3,2e3], stopband=[0.5e3,3e3], max_passband_loss=2, min_stopband_attenuation=15,btype='bandpass')
myfilter3=dict(passband=[2e3,4e3], stopband=[1e3,6e3], max_passband_loss=2, min_stopband_attenuation=15,btype='bandpass')
myfilter15=dict(passband=[10e3,20e3], stopband=[5e3,30e3], max_passband_loss=2, min_stopband_attenuation=15,btype='bandpass')
# taper filter takes 13 sec cf 6.5 sec without for 17 channels on shot 31169
myfilterN = dict(centre=30e3,bw=1e3,taper=2)

filter = None  
help=0
separate=1
closed=True
offset=0   # the approx delta phase expected (helps check aliasing)

verbose=0
max_fs = 2
shot_number = None

exec(pf.utils.process_cmd_line_args())
if help==1: 
    print(__doc__) 
    exit()

#dev_name='LHD'
if dev_name == 'LHD': 
    if diag_name == '': diag_name= 'MP2010'
    if shot_number is None: shot_number = 27233
    #shot_range = range(90090, 90110)
elif dev_name.find('H1')>=0: 
    if diag_name == '': diag_name = "H1DTacqAxial"
    if shot_number is None: shot_number = 69270


device = pf.getDevice(dev_name)

try:
    shot_cache
except:
    print('shot cache not available - use run -i next time to enable')
    shot_cache = {}


print(" %s using getch" % (['not', 'yes, '][use_getch]))
if use_getch: print('plots most likely will be suppressed - sad!')
else: print('single letter commands need to be followed by a CR')

this_key = "{s}:{d}".format(s=shot_number, d=diag_name)
if shot_cache.has_key(this_key): # we can expect the variables to be still around, run with -i
    d = deepcopy(shot_cache[this_key])
else:
    print('get data for {k}'.format(k=this_key))
    d = device.acq.getdata(shot_number, diag_name) # ~ 50MB for 6ch 1MS. (27233MP)
    shot_cache.update({this_key: deepcopy(d)})

if time_range != None:
    d = d.reduce_time(time_range, fftopt=True)  # could use d.reduce_time(copy=False,time_range)

if lowpass != None: 
    if highpass is None:
        d = d.sp_filter_butterworth_bandpass(
            lowpass*1e3,lowpass*2e3,2,20,btype='lowpass')
    else:
        bp = [1e3*lowpass,1e3*highpass]
        bs = [0.5e3*lowpass,1.5e3*highpass]
        d = d.sp_filter_butterworth_bandpass(bp, bs,2,20,btype='bandpass')
elif filter != None:
    if filter.has_key('btype'):
        d = d.sp_filter_butterworth_bandpass(**filter)
    else:
        (fc,df) = (filter['centre'],filter['bw']/2.)
        d = d.filter_fourier_bandpass(
            passband=[fc-df,fc+df], stopband = [fc-2*df, fc+2*df], 
            taper = filter['taper'])
else:
    pass # no filter

if start_time is None:
    sv = d.svd()
    sv.svdplot(hold=hold)

else:
    # first copy the segments into a list, so they can be addressed
    # this doesn't seem to take up much extra memory.
    segs=[]
    for seg in d.segment(numpts):
        segs.append(seg)
    starts = [seg.timebase[0] for seg in segs]
    ord_segs=[]
#    for ii in argsort(starts):
    i=0
    argsrt = argsort(starts)
    while i < len(starts):
        ii = argsrt[i]
        seg=segs[ii]
        if seg.timebase[0] > start_time: 
#            print("normalise = %s" % normalise)
            if (normalise != 0) and (normalise != '0'): 
# the old code used to change seg, even at the beginning of a long chain.
                seg_proc=seg.subtract_mean().normalise(normalise,separate)
                outsvd=seg_proc.svd()
                outsvd.svdplot(hold=hold)
            else: 
                seg_proc=seg.subtract_mean()
                outsvd=seg_proc.svd()
                outsvd.svdplot(hold=hold)
            try: # plotting mag
                if verbose: print(outsvd.history)
                if plot_mag and (seg_proc.scales != None):
                    fig=pl.gcf()
                    oldtop=fig.subplotpars.top
                    fig.subplots_adjust(top=0.65)
                    ax = pl.axes([0.63,0.75,0.35,0.15])
#                    ax=pl.subplot(8,2,-2) # try to put plot on top: doesn't work in new version
                    xticks = range(len(seg_proc.scales))
                    if pyfusion.VERBOSE>3: print('scales',len(seg_proc.scales),seg_proc.scales)
                    pl.bar(xticks, seg_proc.scales, align='center')
                    ax.set_xticks(xticks)
                    # still confused - sometimes the channels are the names bdb
                    try:
                        seg_proc.channels[0].name
                        names = [sgch.name for sgch in seg_proc.channels]
                    except:
                        names = seg_proc.channels

                    try:
                        phi = np.array([float(pyfusion.config.get
                                              ('Diagnostic:{cn}'.
                                               format(cn=c.name), 
                                               'Coords_reduced')
                                              .split(',')[0]) 
                                        for c in seg.channels])
                    except:
                        print('no phi values found')
                        phi = np.arange(len(seg.channels))

                    short_names,p,s = pf.data.plots.split_names(names)
                    #short_names[0]="\n"+seg_proc.channels[0].name  # first one in full
                    pl.xlabel('svd:' +seg_proc.channels[0].name)  # first one in full
                    ax.set_xticklabels(short_names)
                    ax.set_yticks(ax.get_ylim())
# restoring it cancels the visible effect - if we restore, it should be on exit
#                    fig.subplots_adjust(top=oldtop)
            except None:        
                pass
            pl.suptitle("Shot %s, %s t_mid=%.5g, norm=%s, sep=%d" % 
                        (shot_number, diag_name, average(seg_proc.timebase),
                         normalise, separate))
            # now group in flucstrucs - Note: already normalised, so method=0
            fs_set=seg_proc.flucstruc(method=0, separate=separate)
            fs_arr = order_fs(fs_set)
            for fs in fs_arr[0:min([len(fs_arr)-1,max_fs])]:
                RMS_scale=sqrt(mean(seg_proc.scales**2))
                print("  amp=%.3g:" % (sqrt(fs.p)*RMS_scale)),

                print("f=%.3gkHz, t=%.3g, p=%.2f, a12=%.2f, E=%.2g, adjE=%.2g, %s" %
                      (fs.freq/1000, fs.t0, fs.p, fs.a12, fs.E,
                       fs.p*fs.E*RMS_scale**2, fs.svs()))
            if plot_phase:
                # first fs?        
                fs=fs_arr[0]    
                ax = pl.axes([0.21,0.75,0.35,0.15])
    #            ax=pl.subplot(8,2,-3)
                fs.fsplot_phase(closed=closed,offset=offset)    
                pl.xlabel('fs_phase')
                pl.ylim([-4,4])
                #end if plotmsg

            hold = 0 
            print('resetting hold to 0') # for the sake of checkbuttons
            if use_getch: 
                pl.show()
                k=getch.getch()
            else: k=raw_input('enter one of "npqegshS?" (return->next time segment)')
            if k=='': k='n'
            if k in 'bBpP': i-=1
            # Note - if normalise or separate is toggled, it doesn't
            #    affect segs already done.
            elif k=='?': print('s[signals in new frame], h[toggle hold], q[quit] b/p[back] S[toggle separate]')
            elif k in 'tT': 
                if (normalise==0) or (normalise=='0'): normalise ='rms'
            elif k in 'qQeE':i=999999
            elif k in 'gG': use_getch=not(use_getch)
            elif k in 'S': separate=not(separate)
            elif k in 'h': 
                hold=not(hold)
                if hold: pl.figure() # make a new figure so that we don't overwrite the old.
                else: pass 
            elif k in 's': # plot the signals in a new frame
                pl.figure()
                segs[ii].plot_signals()
                pl.figure(1)
            else:  i+=1
            if verbose: print i,ii
        else: i+=1

        
