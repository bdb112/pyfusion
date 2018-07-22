""" walk through the examples folder and run them according to comments contained - if any
e.g.
_PYFUSION_TEST_@@diag_name="DIA135"
Highly recommended to run in a VNC desktop so that the focus is not stolen on each test
  -- so far, with xfce, should use xterm to allow copy (assuming vncconf is up

To test an old checkout with the extra files needed
clone_pyfusion  # ( an alias)
source /tmp/testpf/pyfusion/pyfusion/run_pyfusion 57 Lim57%1
cd ~/pyfusion/working/pyfusion
run /home/bdb112/pyfusion/working/pyfusion/pyfusion/test/test_examples.py newest_first=0 filewild="/tmp/testpf/pyfusion/pyfusion/examples/*py"

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
start = 0  # allow a restart part-way through - always starts earlier, as it ignores @@Skip
#       start=last_one is got short_cut
pfdebug=0 # normally set pyfusion.DEBUG to 0 regardless
newest_first=1 # if True, order the files so the last edited is first.
max_sec=2
stop_on_error=False
maxwidth=90 # was 77  # maximum width of error message in summary display
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

###  THIS Needs to happen through the env! 
os.environ['PYFUSION_DEBUG'] = str(pfdebug)

os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':/home/bdb112/python'

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
        gitshort = [os.path.join(os.path.split(gf.decode())[-1]) for gf in gitlist]
    else:
        gitlist = None    
except Exception as reason:
    print('exception in git call', reason)
    gitlist = None

tm = localtime()
wildlist = glob.glob(filewild)

if start  == 0:
    out_list, err_files = [], []
    total = 0

if gitlist is not None:
    filelist = [wf for wf in wildlist if os.path.join(os.path.split(wf)[-1]) in gitshort]
    if (len(filelist) == 0) and len(wildlist)>0:
        print('******** no matching files in the git repo ******* \n{f} etc'
              .format(f=wildlist[0:5]))
else:
    filelist = wildlist

# use alphabetical order as a reference
alpha_order = np.argsort([os.path.split(f)[-1] for f in filelist])
filelist = [filelist[i] for i in alpha_order]
if len(filelist) == 0:
    raise LookupError('No files found matching {wi}'.format(wi=filewild))
print('First alphabetically is ' + filelist[0] + '\n')
orig_order = np.arange(0, len(filelist))

if newest_first:
    order = np.argsort([os.path.getmtime(f) for f in filelist])
    filelist = np.array(filelist)[order[::-1]]
    orig_order = orig_order[order[::-1]]

try:  # this try is to catch ^C
    for this_one, filename in enumerate(filelist[start:]):  # [1:3] for test
        #  print(this_one, filename)
        prerun, tmpfil = '', ''
        flags = look_for_flags(filename)
        args = ''
        print('flags are', flags)
        if flags != []:
            if ('skip' in [flag.lower() for flag in flags] or
                'script' in [flag.lower() for flag in flags]):
                # this is fudgey to get the number of elements right
                flag = ','.join([flag for flag in flags
                                  if 'skip' in flag.lower() or 'script' in flag.lower()])
                out_list.append([filename, flag.lower().encode(), b'Skipping', 0, 0])
                continue  # this stops it from further consideration (and from being run)
            else:
                for flag in flags:
                    if "PRE@" in flag:
                        prerun = flag.split('PRE@')[1]
                    elif '=' in flag:
                        args += ' '+flag
                    elif np.any([tok in flag.upper() for tok in 'NOTSKIP,PYFUSION_TEST'.split(',')]):
                        pass
                    else:
                        args += ' '+flag # probably the first arg
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
        have_error = (((errout != b'')
                       and (not b'warn' in errout.lower()))
                      or (sub_pipe.returncode != 0))
        if have_error:
            err_files.append([this_one + start, filename])

        print(resp[-10000:])
        dt = round(seconds() - st, 3)
        out_list.append([filename, errout, resp, dt, sub_pipe.returncode])
        total += 1
        if have_error and stop_on_error:
            break

except KeyboardInterrupt as reason:
    print('KeyboardInterrupt', reason)

except Exception as reason:
    print('exception during execution ', reason)

dumpname = str('test_output_V{V}_{yy:02d}{mm:02d}{dd:02d}_{hh:02d}:{mn:02d}.pickle'
               .format(yy=tm.tm_year-2000, mm=tm.tm_mon, dd=tm.tm_mday,
                       hh=tm.tm_hour, mn=tm.tm_min,
                       V=sys.version[0:5]))

pickle.dump(out_list, open(dumpname, 'wb'))
last_one = this_one + start

print()
print('Python {pv}, Pyfusion {pfv} {date}'.format(pv=sys.version[0:20], pfv=pyfusion.VERSION, date=ctime()))
for i, ll in enumerate(out_list):
    exception_lines = list({lin[0:maxwidth] for lin in ll[1].split(b'\n') if b'Error:' in lin})  # set comprehension elim. dups
    elines = b' '.join(exception_lines) if len(exception_lines) > 0 else ll[1]
    elines = elines + ll[1].split(exception_lines[0])[1][0:maxwidth-len(elines)] if len(elines) < maxwidth and len(exception_lines)>0 else elines
    print('{o:03d} {dt:5.2f} {fn:30s}: {msg}'
          .format(o=orig_order[i], dt=ll[3], fn='/'.join(ll[0].split('/')[-2:]),
                  msg=[b'Err ', b'OK! '][ll[-1] == 0] + [elines[-maxwidth:].replace(b'\n', b' ')][0]))

print('{g} good, {e} errors out of {t} not skipped'.format(e=len(err_files), t=total, g=total-len(err_files)))

if '-3' in python_exe:
    print('python 3 warnings coming from my files')
    for (n, ll) in enumerate(out_list):
        if 'bdb112' in ll[1]:
            print(n, ll[0])

print('to see problem files, type print(err_files)')
"""
put this at the end of the file written
%%% Local Variables: 
%%% fill-column: 999999 
%%% End: 
"""
