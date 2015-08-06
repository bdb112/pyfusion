#!/usr/bin/env python
""" 
   !Warning - don't use run -i!  - will get pickle errors.
   Also, if there is an input error, it just waits for input....

Like prepfs_range, but MP.  initially spawn one process for each time interval, 
maybe later do multiples - however pool.map wants to do one shot
per process.
This code uses multi threading, giving each thread a range of shots
one by one. (but I think this implemetation (multiprocess)uses processes 
instead of threads - I used this as it seems more widely used.)
Maybe using the "multiprocess.dummy" I can use bare thareds, but seems not
nexessary.

The threads then spawn a process for each shot - this seems wasteful,
but the thread is very cheap, and the spawned subprocess serves to detach
the pyfusion calculations, so that errors are less likely to break the
processing of multiple shots.
Job controller for preprocessing pyfusion flucstruc data.  Replaces 
do_range script, so that a nice argument form can be used, and eventually 
multi-processing.

Heliotron:

Note: quoting!! escape with \'\"  and \"\'
python pyfusion/examples/prepfs_time_mp.py . --time_range='[0,.1]' --MP=4 --shot=87200 --dev_name='H1Local' --diag_name="H1ToroidalAxial"
#complicated
wait_for_MDS_data tree=mirnov shot=${sht} path='.ACQ132_8:INPUT_01' && python pyfusion/examples/prepfs_time_mp.py . --exe='gen_fs_bands.py n_samples=None df=1e3  max_bands=3 dev_name=H1Local' --shot=[${sht}] --diag_name="H1ToroidalAxial" --overlap=2.5 --exception=Exception --debug=0 --seg_dt=0.0005 --time_range=[0,0.06]
"""
from warnings import warn
import numpy as np
import subprocess
import os

def worker(arglist): 
    from time import localtime
    os.nice(3)   # seems to have a bigger effect on nice index?
    time_range,ARGS = arglist
    print('worker on ', time_range , ARGS, arglist)
    tm = localtime()

    sr=ARGS.shot   # in this mp version, sr will always be a list of single shot 
                      
    filename = ARGS.filename_format\
        .format(sr=ARGS.shot, nor=ARGS.normalize, sep=ARGS.separate, 
                dn = ARGS.diag_name, ns=ARGS.n_samples,
                tb=int(1000*time_range[0]), te=int(1000*time_range[1]), 
                yy=tm.tm_year-2000,mm=tm.tm_mon,tr=time_range,
                dd=tm.tm_mday,hh=tm.tm_hour
                )
    if ARGS.output_path[0] in [None, "None", ""]:
        redir = ""
        outd = ""
    else:
        redir = str(' > {path}/{fn}'
                    .format(path=ARGS.output_path[0],fn=filename))
        outd = ARGS.output_path[0]

    cmd = str('python pyfusion/examples/{exe} shot_range=[{sh}] diag_name={dn} '
              'overlap={ov} exception={ex} debug={db} dev_name={dev_name} '
              '  n_samples={ns} seg_dt={seg_dt:.4g} time_range=[{tb},{te}] '
              'separate={sep} info={info} method="{nor}" {redir}'
              .format(exe=ARGS.exe,sh=ARGS.shot, 
                      nor=ARGS.normalize, sep=ARGS.separate, 
                      dn = ARGS.diag_name, db=ARGS.debug,
                      tb=time_range[0], te=time_range[1],
                      ex=ARGS.exception, ov=ARGS.overlap, ns=ARGS.n_samples,
                      seg_dt=ARGS.seg_dt,
                      redir=redir, info=ARGS.info,
                      dev_name=ARGS.dev_name
                      )
              )

    print(cmd)

    sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    (resp,err) = sub_pipe.communicate()
    if (err != '') or (sub_pipe.returncode != 0): 
        print(resp,err,'.') #
        print(err.split('\n')[-2]),
    if resp !='': print(resp[-10:])
    #sub_pipe.terminate()  # too harsh
    print(' {tr} {s} done!'.format(s=ARGS.shot,tr=time_range))
    return(err)

def mymap(pool, fn, list, extra):
    import itertools
    zippedarg = itertools.izip(list, itertools.repeat(extra))
    return(pool.map(fn, zippedarg))

if __name__ == '__main__':

    try:
        from argparse import ArgumentParser, FileType
    except ImportError:   # may work better in early versions
        warn("can't find argparse - trying Ipython version instead")
        import IPython
        from IPython.external.argparse import ArgumentParser, FileType


    PARSER = ArgumentParser( description='job controller for preprocessing pyfusion data.: '+__doc__ )
    # note - default= for positional arg requires nargs = '?' or '*'
    PARSER.add_argument( 'output_path', metavar='output_path', help=' path to store output text files', default='.', nargs=1, type=str) # just 1 for now
    PARSER.add_argument( '--filename_format', type=str, 
                         default='PF2_{yy:02d}{mm:02d}{dd:02d}_{dn}_{sr}_{tb}_{te}_{ns}_{nor}_{sep}', 
                         help='format string to produce output file name' )
    PARSER.add_argument( '--exe', type=str, 
                         default='gen_fs_bands.py', 
                         help='script to run - can include extra args for'\
                             'the exe line here too' )
    PARSER.add_argument( '--debug', type=int, default=0, 
                         help='debug level, > 1 will prevent exceptions being hidden - useful only an manual runs, not with prepfs_range')
    PARSER.add_argument( '--info', type=int, default=1, 
                         help='controls how much history and config info is printed')
    PARSER.add_argument( '--quiet', type=int, default=0, 
                         help=' >0 suppresses printout of extra information' )
    PARSER.add_argument( '--separate', type=int, default=1, 
                         help='0 - normalise all probes by the same factor - 1 => separately' )
    PARSER.add_argument( '--MP', type=int, default=1, 
                         help='number of processes to use' )
    PARSER.add_argument( '--n_samples', type=int, default=None, 
                         help='number of samples in FFT' )
    PARSER.add_argument( '--overlap', type=float, default=2, 
                         help='overlaps of FFT intervals - 0 -> none, 1 ~ 50%' )
    PARSER.add_argument( '--shot', type=str, default='54185', 
                         help='shot - an integer' )
    PARSER.add_argument( '--normalize', type=str, default='rms', 
                         help="normalisation function - e.g. 'rms' or '0' for not, or 'None' for default" )
    PARSER.add_argument( '--time_range', type=str, default='None', 
                         help="range of times for analysis e.g. [.1,2] or normally 'None' for all" )
    PARSER.add_argument( '--seg_dt', type=float, default=1.5e-3, 
                         help="time interval for STFFT typically 1.5e-3" )
    PARSER.add_argument( '--exception', type=str, default='Exception', 
                         help="type of exceptions to be ignored - set to None for debugging" )
    PARSER.add_argument( '--diag_name', type=str, default='MP2010', 
                         help="Mirnov probe set for analysis - e.g. MP2010: see .cfg" )
    PARSER.add_argument( '--dev_name', type=str, default='H1Local', 
                         help="Mirnov probe set for analysis - e.g. MP2010: see .cfg" )

    ARGS = PARSER.parse_args()
    time_range = eval(ARGS.time_range)
    if len(np.shape(time_range)) == 1:
        if ARGS.MP>0:  # split if MP and only 1 range
            times = np.linspace(time_range[0],time_range[1],ARGS.MP+1)
            time_range = [[times[i],times[i+1]] for i in range(ARGS.MP)]
        else:
            time_range=[time_range]

    os.nice(1)   # helps identify the leader - but decrs every time you run
    if ARGS.MP>0:
        import multiprocessing, itertools
        pool = multiprocessing.Pool(ARGS.MP)

        #results = pool.map(worker, shot_range)#, ARGS)#, itertools.repeat(ARGS))
        results = mymap(pool, worker, time_range, ARGS)
    else:
        results = []
        for time in time_range:
            results += worker((time, ARGS))


    import pickle
    import time as tm

    p=open(tm.strftime(ARGS.output_path[0]+'/%Y%m%d%H%M%S_prepfs_data.pickle'),'w')
    pickle.dump(results,p)
    p.close()







