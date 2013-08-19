#!/usr/bin/env python
""" 
Like prepfs_range, but separate jobs and files for each shot
job controller for preprocessing pyfusion flucstruc data.  Replaces 
do_range script, so that a nice argument form can be used, and eventually 
multi-processing.
"""
from warnings import warn
import subprocess


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
                         default='PF2_{yy:02d}{mm:02d}{dd:02d}_{dn}_{sr}_{ns}_{nor}_{sep}', 
                         help='format string to produce output file name' )
    PARSER.add_argument( '--exe', type=str, 
                         default='gen_fs_local.py', 
                         help='script to run - can include extra args for'\
                             'the exe line here too' )
    PARSER.add_argument( '--debug', type=int, default=0, 
                         help='debug level, > 1 will prevent exceptions being hidden - useful only an manual runs, not with prepfs_range')
    PARSER.add_argument( '--info', type=int, default=2, 
                         help='controls how much history and config info is printed')
    PARSER.add_argument( '--quiet', type=int, default=0, 
                         help=' >0 suppresses printout of extra information' )
    PARSER.add_argument( '--separate', type=int, default=1, 
                         help='0 - normalise all probes by the same factor - 1 => separately' )
    PARSER.add_argument( '--n_samples', type=int, default=None, 
                         help='number of samples in FFT' )
    PARSER.add_argument( '--overlap', type=float, default=2, 
                         help='overlaps of FFT intervals - 0 -> none, 1 ~ 50%' )
    PARSER.add_argument( '--shot_range', type=str, default='[54185]', 
                         help='range of shots - e.g. [1,2,3] or range(1,4) etc' )
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

    ARGS = PARSER.parse_args()

    from time import localtime
    tm = localtime()

    shot_range = eval(ARGS.shot_range)
    #if len(shot_range)==1: sr = shot_range[0]
    #else: sr = "{min}_{max}".format(min=min(shot_range), max=max(shot_range))
    for shot in shot_range:
        sr=shot
        filename = ARGS.filename_format.format(sr=sr, nor=ARGS.normalize, sep=ARGS.separate, 
                                               dn = ARGS.diag_name, ns=ARGS.n_samples,
                                               yy=tm.tm_year-2000,mm=tm.tm_mon,dd=tm.tm_mday,hh=tm.tm_hour
                                               )

        cmd = str('python pyfusion/examples/{exe} shot_range=[{sr}] diag_name={dn} '
                  'overlap={ov} exception={ex} debug={db} '
                  '  n_samples={ns} seg_dt={seg_dt:.4g} time_range={tr} '
                  'separate={sep} info={info} method="{nor}" > {path}/{fn}'
                  .format(exe=ARGS.exe,sr=shot, 
                          nor=ARGS.normalize, sep=ARGS.separate, 
                          dn = ARGS.diag_name, tr=ARGS.time_range, db=ARGS.debug,
                          ex=ARGS.exception, ov=ARGS.overlap, ns=ARGS.n_samples,
                          seg_dt=ARGS.seg_dt,
                          path=ARGS.output_path[0],fn=filename, info=ARGS.info
                          )
                  )

        print(cmd)

        sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        (resp,err) = sub_pipe.communicate()
        if (err != '') or (sub_pipe.returncode != 0): 
            print(resp,err,'.') #
        print(resp[-10000:])
        #sub_pipe.terminate()  # too harsh
    print('Done')
