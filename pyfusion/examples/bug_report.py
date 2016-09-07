""" generate output to aid in debugging
Under the bash shell, it can all be piped to a file.
python pyfusion/examples/bug_report.py >& /tmp/elog 

you can also attempt to run a script that needs no arguments
e.g.
run pyfusion/examples/bug_report.py pyfusion/examples/JSPF_tutorial/example3.py
"""
import sys
import os
print(sys.version)
print(os.uname())
try:
    import pyfusion
    print('pyfusion version {v}'.format(v=pyfusion.VERSION))
except Exception as reason:
    print('\nError importing pyfusion {e}'.format(e=reason))
    raise

if len(sys.argv) > 1:
    fname = sys.argv[1]
    # this clumsy line works in python2 and 3
    exec(compile(open(fname).read(), fname, 'exec'))
