""" walk through the examples folder and run them according to comments contained - if any
e.g.
_PYFUSION_TEST_@@diag_name="DIA135"
""" 


from __future__ import print_function
import subprocess, glob, pickle, sys
import numpy as np
from time import localtime, ctime, sleep
import tempfile, os
import pyfusion
from time import time as seconds

_var_defaults="""
filewild = 'pyfusion/examples/*.py'  # an explict filewild (but gets non repo)
python_exe = 'python' # -3'
start = 0  # allow a restart part-way through - always early, as it ignores @@Skip
pfdebug=0 # normally set pyfsion.DEBUG to 0 regardless
newest_first=1 # if True, order the files so the last edited is first.
max_sec=2
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

###  THIS Needs to happen through the env! 
os.environ['PYFUSION_DEBUG'] = str(pfdebug)

def look_for_flags(filename):
    with open(filename) as f:
        buff=f.readlines()
    toks = []
    for lin in buff:
        if '_PYFUSION_TEST_' in lin:
            toks = lin.strip().split('@@')

    return(toks)

try:
    cmd = 'git ls-files'
    sub_pipe = subprocess.Popen(cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (gitresp, giterrout) = sub_pipe.communicate()
    print(giterrout)
    if sub_pipe.returncode == 0:
        #gitlist = [os.path.join(os.getcwd(), pth) for pth in gitresp.split()]
        gitlist = gitresp.split()
        gitshort = [os.path.join(os.path.split(gf)[-1]) for gf in gitlist]
    else:
        gitlist = None    
except Exception as reason:
    print('exception in git call', reason)
    gitlist = None

tm = localtime()
wildlist = glob.glob(filewild)

if start  == 0:
    out_list = []
    n_errs, total = 0, 0

if gitlist is not None:
    filelist = [wf for wf in wildlist if os.path.join(os.path.split(wf)[-1]) in gitshort]
    if (len(filelist) == 0) and len(wildlist)>0:
        print('******** no matching files in the git repo ******* \n{f} etc'
              .format(f=wildlist[0:5]))
else:
    filelist = wildlist

orig_order = np.arange(0, len(filelist))
if newest_first:
    order = np.argsort([os.path.getmtime(f) for f in filelist])
    filelist = np.array(filelist)[order[::-1]]
    orig_order = orig_order[order[::-1]]

try:  # this try is to catch ^C
    for filename in filelist[start:]:  # [1:3] for test
        prerun, tmpfil = '', ''
        flags = look_for_flags(filename)
        args = ''
        print(flags)
        if flags != []:
            if 'skip' in [flag.lower() for flag in flags]:
                continue  # this stops it from further consideration (and from being run)
            else:
                for flag in flags:
                    if "PRE@" in flag:
                        prerun = flag.split('PRE@')[1]
                    elif '=' in flag:
                        args += ' '+flag
                print('run with', args)

# need to cd for the JSPF examples:
        if '/JSPF_tut' in filename:
            prerun = '\n'.join([prerun, 'import os', 'os.chdir("pyfusion/examples/JSPF_tutorial/")'])

        if prerun != '':  # if we need to do somthing special - e.g. env vars
            runfile = tempfile.mktemp()
            env = os.environ
            # env.update({'PYTHONSTARTUP':tmpfil}) only works in interactive
            with open(filename, 'rt') as pf:
                prog = pf.readlines()
            with open(runfile, 'wt') as tf:
                tf.write(prerun+'\n')
                tf.writelines(prog)
        else:
            env = None
            runfile = filename

        cmd = '{py} {file} {args}'.format(file=runfile, py=python_exe, args=args)
        print(cmd)

        st = seconds()
        sub_pipe = subprocess.Popen(cmd,  env=env, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        """ Doesn't workthis way - seems to wait.
        for t in range(max_sec):
            print(t)
            sleep(1)
            if sub_pipe.poll():
                break
            if t == max_sec - 1:
                print('terminate')
                sub_pipe.kill()
        """
        (resp, errout) = sub_pipe.communicate()
        print(errout)
        if ((errout != b'') and (not 'warn' in errout.lower())) or (sub_pipe.returncode != 0):
            print(resp, errout, '.')
            n_errs += 1

        print(resp[-10000:])
        dt = round(seconds() - st, 3)
        out_list.append([filename, errout, resp, dt])
        total += 1
except KeyboardInterrupt as reason:
    print('KeyboardInterrupt', reason)

except Exception as reason:
    print('exception during execution ', reason)

dumpname = str('test_output_V{V}_{yy:02d}{mm:02d}{dd:02d}_{hh:02d}:{mn:02d}.pickle'
               .format(yy=tm.tm_year-2000, mm=tm.tm_mon, dd=tm.tm_mday,
                       hh=tm.tm_hour, mn=tm.tm_min,
                       V=sys.version[0:5]))

pickle.dump(out_list, open(dumpname, 'wb'))

print()
print('Python {pv}, Pyfusion {pfv} {date}'.format(pv=sys.version[0:20], pfv=pyfusion.VERSION, date=ctime()))
for i, ll in enumerate(out_list):
    print('{o:3d} {dt:5.2f} {fn:30s}: {msg}'
          .format(o=orig_order[i], dt=ll[3], fn='/'.join(ll[0].split('/')[-2:]),
                  msg=[ll[1][-57:].replace(b'\n', b' '), b'OK!'][ll[1] == b'']))

print('{g} good, {e} errors out of {t}'.format(e=n_errs, t=total, g=total-n_errs))

if '-3' in python_exe:
    print('python 3 warnings coming from my files')
    for (n, ll) in enumerate(out_list):
        if 'bdb112' in ll[1]:
            print(n, ll[0])
"""
put this at the end of the file written
%%% Local Variables: 
%%% fill-column: 999999 
%%% End: 
"""
