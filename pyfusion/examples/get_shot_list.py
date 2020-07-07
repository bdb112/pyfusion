""" Return a shot_list like  [[20180919, 45], [20180919, 50], [20180919, 55]] in shot_list
    Example:
     run pyfusion/examples/get_shot_list.py
     run pyfusion/examples/get_shot_list.py path='/data/datamining/local_data/W7X/short'

    Don't forget to run -i in the next step.
    Returned as a list, so that comparison can be done without worrying about all()

   Note: requires more effort to run on windows - need to invoke bash shell
   Print the command line if the command fails, and wait for the file.

   Really better to do all this in python....
"""
import numpy as np
import os

from pyfusion.utils import process_cmd_line_args

_var_defaults="""
path = "/data/datamining/local_data/W7X"
diag = "MIR"
"""

exec(_var_defaults)
exec(process_cmd_line_args())

cmd = str('pyfusion/bin/get_W7X_shotnums {path} {diag} > /tmp/file_list'
          .format(**locals()))
extra_shell = '' if os.name == 'posix' else 'c:\\cygwin\\bin\\bash.exe -c '
retcode = os.system(extra_shell + cmd)
# in test_examples.py this somehow comes up as an error
if retcode != 0:
    print('failed with code {retcode} executing \n{cmd}'.format(**locals()))
    x=input('execute manually and run again to read data, then hit return')

shot_list = np.loadtxt('/tmp/file_list', delimiter=',',dtype='int').tolist()
print('Retrieved {num} shot/diag combinations from {days} days'
      .format(num=len(shot_list), days=len(np.unique([s[0] for s in shot_list]))))
