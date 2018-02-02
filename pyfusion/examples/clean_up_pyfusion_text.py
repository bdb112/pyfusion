"""
strip bad lines out of pyfusion text files.

Seems 2x slower in ana 3 and 2.7.10 (9sec) than 2.7.6 (4secs) PF2_130812_MP2010HMPno612_65139_384_rms_1

(sort(lines[next_good:])==sort(lines[2*next_good-num-1:next_good-1])).all()

Note the funny quotes below
(should choose a few samples files, one that run faster)
_PYFUSION_TEST_@@fileglob="glob.glob('PF2*1')[0]"
"""
from six.moves import input
import bz2
import os
import numpy as np
import sys
if sys.version>'3,':
    import io 
    print('Warning - this runs much slow with the io module - probably needs a rewrite')
else:
    import StringIO as io
import glob

def my_move(filename, subfolder='bad',overwrite=False):
    """ move a file to a subfolder, making it if required
    """
    (this_folder, fn) = os.path.split(filename)
    todir = os.path.join(this_folder, subfolder)

    if not os.path.isdir(todir):  os.mkdir(todir)
    newfullname = os.path.join(todir, fn)
    while (not overwrite) and os.path.isfile(newfullname):
        newfullname += '.tmp'
    os.rename(filename, newfullname)
    return(newfullname)


_var_defaults="""
debug=1
all_errors=[]
baddir = 'bad'
fileglob = None  #  None alloes a glob expression to be entered
fast = False

"""

exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())




#filename='/home/bdb112/datamining/preproc/PF2_121208/PF2_121209_MP2010_99931_99931_1_384_rms_1.dat.bz2'  # I killed it
#filename = '/home/bdb112/datamining/preproc/PF2_121208/foo.bz2'
#for filename in np.sort(glob.glob('/home/bdb112/datamining/preproc/PF2_121208/*2')):
#for filename in np.sort(glob.glob('/home/bdb112/datamining/preproc/PF2_121208/PF2_121209_MP2010_113431_113431_1_384_rms_1.dat.bz2')):  # I killed it
good_lines = 0

if fileglob is None:
    fileglob = '/mythextra/datamining/PF2_121208/*2'

for filename in np.sort(glob.glob(fileglob)):

    ext = os.path.splitext(filename)[1]
    if ext == 'gz':
        FileReader = gzip.open
    elif ext == 'bz2':
        FileReader = bz2.BZ2File
    else:
        FileReader = open

    with FileReader(filename) as fd:
        lines=fd.readlines()

      # gets the first occurrence of 'Shot'
    shotline_candidates = np.where(np.char.find(lines,'Shot')==0)[0]
    if len(shotline_candidates)==0: 
        print('no Shot line in '+filename)
        all_errors.append([filename, len(lines), [-1], ['no shotline']]) # -1 means no shotline
        print('moving to '+ my_move(filename, 'no_shot_line'))
        continue
    if len(shotline_candidates)!=1: 
        print('Extra shot line in '+filename, ' using last')

    shotline = shotline_candidates[-1]
    if shotline == (len(lines) - 1):
        print('no fs data in '+filename)
        all_errors.append([filename, len(lines), [-2], ['no fs data']]) # -2 means no data
        print('moving to ' +my_move(filename,'no_fs'))
        continue
        
    shotlinetext = lines[shotline]
    header_toks = shotlinetext.split()

    ph_dtype = None
    f='f8'
    if ph_dtype is None:
        ph_dtype = [('p12',f),('p23',f),('p34',f),('p45',f),('p56',f)]
        #ph_dtype = [('p12',f)]


    # is the first character of the 2nd last a digit?
    if header_toks[-2][0] in '0123456789': 
        if pyfusion.VERBOSE > 0: 
            print('found new header including number of phases')
        n_phases = int(header_toks[-2])
        ph_dtype = [('p{n}{np1}'.format(n=n, np1=n+1), f) for n in range(n_phases)]

    if 'frlow' in header_toks:  # add the two extra fields
        fs_dtype= [ ('shot','i8'), ('t_mid','f8'), 
                    ('_binary_svs','f8'), #'i8'), OverflowError: long too big to convert
                    ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                    ('p', 'f8'), ('H','f8'), 
                    ('frlow','f8'), ('frhigh', 'f8'),('phases',ph_dtype)]
    else:
        fs_dtype= [ ('shot','i8'), ('t_mid','f8'), 
                    ('_binary_svs','f8'),  # 'i8'), 
                    ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                    ('p', 'f8'), ('H','f8'), ('phases',ph_dtype)]
    errors = []
    badlines = []

    try:
        dat = np.loadtxt(io.StringIO(''.join(lines[shotline+1:])), 
                         dtype=fs_dtype, ndmin=1)
        if debug: print('seems OK'),
        fast_read_error = 0
        good_lines += len(dat['shot'])

    except (IndexError, ValueError):
        fast_read_error = 1

    if not fast or fast_read_error > 0:
        print('{p} errors in {f}'.format(p=['checking for','processing'][fast_read_error], f=filename))
        num = len(lines)
        # try one line at a time
        this_shot = None
        last_time = None
        for l in range(shotline+1, num)[::-1]:  # read backwards so we can pop
            try:
                dat = np.loadtxt(io.StringIO(lines[l]), dtype=fs_dtype, ndmin=1)
                read_error = 0
                if this_shot is None:
                    this_shot = dat['shot']
                else:  # note - only catches this error if not "fast" or the 
                    # shot error could be a corrupted line, but also might be another shot?
                    if this_shot != dat['shot']: #global read fails 
                        read_error = 2           #(i.e. must be another error)    

                if last_time is None:
                    last_time = dat['t_mid']
                else:  # note - only catches this error if not "fast" etc as above
                    #print(last_time, dat['t_mid'])
                    if last_time < dat['t_mid']: # reading backwards, so < !!!
                        read_error += 4          # a different error code
                    last_time = dat['t_mid']

            except:
                read_error = 1

            if read_error>0:
                errors.append(l)
                badlines.append([l,lines[l]])
                if len(errors)<3: print(lines[l])
                if debug>0: ans = input('delete this line? (Y, A(ll),^C to stop, line{l})'
                                        .format(l=l)).lower()
                if ans=='a': 
                    debug=0
                if ans in ['y','','a']:   # need ans to act!
                    if (read_error & 3): lines.pop(l)    # pop if corrupted (not the other errors)
                    num = len(lines)
                    next_good = l
                    if (((2*next_good-num)>0) and 
                        (np.sort(lines[next_good:])==
                         np.sort(lines[2*next_good-num:next_good])).all()):
                        if debug>0: 
                            print('=== remove {dups} duplicates ==='
                                  .format(dups=num-next_good))
                        errors.append(-(num-next_good))  # negative num( not -1) means dups
                        for ll in range(next_good, num)[::-1]:
                            lines.pop()

                    # remove duplicates from here to end
                    last = len(lines)
                    for lll in range(next_good,last)[::-1]:
                        if (np.char.find(lines[shotline:l], lines[lll])==0).any():
                            lines.pop(lll)
                            errors.append(lll)  # these will be relative to any provious deletes!

                else: sys.exit()

        if len(errors) == 0:
            if debug>0: print('no errors found')
        else:
            renamed = my_move(filename)
            print('errors found in {e}, move to \n{bad}'\
                  ' and rewrite only good lines\n to '
                  .format(e=errors, bad=renamed)),

            newname = filename.rsplit('.',1)[0] # take off the bz2
            # rename old one if there is one  (but not old old one.)
            if os.path.isfile(newname): os.rename(newname,newname + '.old')

            print(newname)
            with file(newname,'w') as fw:
                fw.writelines(lines)
            all_errors.append([filename+' -> '+newname, len(lines), errors, badlines])
            filename = newname  # so that a my_move below moves the right thing

    # if we get here we should be clean!            
    da = np.loadtxt(io.StringIO(''.join(lines[shotline+1:])), 
                    dtype=fs_dtype, ndmin=1)                    
    # len(np.shape to guard against file with just one fs - should this be legal?
    if (len(np.shape(da['t_mid']))>0) and (len(da['t_mid'])>1):
        if (np.min(np.diff(da['t_mid'])))<0: 
            # for now, note but persist with non monotonic times
            biggest_reverse = shotline+2+np.argmin(np.diff(da['t_mid']))
            errors.append(biggest_reverse)
            errors.append(-9999) # means time not mono.
            badlines.append('time not mono') # means time not mono.
            print('time not monotonic at {l} in {f}'.format(l=biggest_reverse, f= filename))
            my_move(filename, 'time_not_mono')
            all_errors.append([filename, len(lines), errors, badlines])
        if not (np.diff(da['shot'])==0).all(): 
            # for now, note but persist with non monotonic times
            one_funny_shot = shotline+2+np.min(np.where(np.diff(da['shot'])!=0)[0])
            errors.append(one_funny_shot)
            errors.append(-88888) # means shot not same
            badlines.append('shot number not same') 
            print(' shot number not same at {l} in {f}'.format(l=one_funny_shot, f= filename))
            all_errors.append([filename, len(lines), errors, badlines])
            my_move(filename, 'bad_shotno')

if good_lines==0:
    print('Warning - no good lines found!')

import pickle
import time as tm

p=open(tm.strftime('%Y%m%d%H%M%S_cleanup_data.pickle'),'wb')
pickle.dump(all_errors,p)

"""

fd=bz2.BZ2File('/home/bdb112/datamining/preproc/PF2_121208/PF2_121209_MP2010_99931_99931_1_384_rms_1.dat.bz2')
problem
'/home/bdb112/datamining/preproc/PF2_121208/PF2_121208_MP2010_56431_56431_1_384_rms_1.dat.bz2'

"""
