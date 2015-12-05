""" walk through the exmaples folder and run them according to comments contained - if any
e.g.
_PYFUSION_TEST_@@diag_name="DIA135"
""" 


from __future__ import print_function
import subprocess, glob, pickle, sys
from time import localtime
_var_defaults="""
filewild = 'pyfusion/examples/*.py'
python_exe = 'python' # -3'
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())


def look_for_flags(filename):
    with open(filename) as f:
        buff=f.readlines()
    toks = []
    for lin in buff:
        if '_PYFUSION_TEST_' in lin:
            toks = lin.strip().split('@@')

    return(toks)

tm = localtime()
out_list = []
filelist = glob.glob(filewild)
n_errs, total = 0, 0
try:
    for filename in filelist:  #[1:3] for test
        cmd = '{py} {file}'.format(file=filename, py=python_exe)
        flags = look_for_flags(filename)
        if flags != []:
            if 'skip' in [flag.lower() for flag in flags]:
                continue
            else:
                for flag in flags:
                    if '=' in flag:
                        cmd += ' '+flag

        print(cmd)

        sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        (resp,err) = sub_pipe.communicate()
        if (err != b'') or (sub_pipe.returncode != 0): 
            print(resp,err,'.') #
            n_errs += 1

        print(resp[-10000:])
        out_list.append([filename, err, resp])
        total += 1
except KeyboardInterrupt:
    pass

dumpname = str('test_output_V{V}_{yy:02d}{mm:02d}{dd:02d}_{hh:02d}:{mn:02d}.pickle'
               .format(yy=tm.tm_year-2000,mm=tm.tm_mon,dd=tm.tm_mday,hh=tm.tm_hour,mn=tm.tm_min,
                       V=sys.version[0:5]))
            
pickle.dump(out_list, open(dumpname,'wb'))

print()
for ll in out_list:
    print('{fn:30s}: {msg}'.format(fn='/'.join(ll[0].split('/')[-2:]),msg=[ll[1][-57:].replace(b'\n',b' '),b'OK!'][ll[1]==b'']))

print('{e} errors out of {t}'.format(e=n_errs, t=total))

if '-3' in python_exe:
    print('python 3 warnings coming from my files')
    for (n,ll) in enumerate(out_list):
        if 'bdb112' in ll[1]:
            print(n, ll[0])




"""
put this at the end of the file written
%%% Local Variables: 
%%% fill-column: 999999 
%%% End: 
"""
