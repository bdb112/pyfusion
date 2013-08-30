import pyfusion
from pyfusion.acquisition.read_text_pyfusion import read_text_pyfusion, merge_ds
from glob import glob
import numpy as np
from pyfusion.debug_ import debug_

_var_default="""
debug=0
target='^Shot .*'  # to allow searches for e.g. '^Shot .*'  or skip lines ==4
quiet=1
append = False
append_old_method =False   # not sure if the old method ever worked - takes a lot of mem
exception=Exception
file_list = [pyfusion.root_dir+'/acquisition/PF2_121206_54185_384_rms_1.dat.bz2']
save_filename=None
"""

exec(_var_default)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if len(np.shape(file_list)) == 0: file_list=[file_list]


(ds_list, comment_list) = read_text_pyfusion(file_list, debug=debug, exception=exception, target=target)

if len(ds_list) == 0: raise LookupError('no files found for {f}'.format(f=file_list))

if append_old_method:
    ds_list.append(dd)
    comment_list.extend(dd['info']['comment'])
if append:
    dd = merge_ds(ds_list, comment_list, old_dd=dd)
else:
    dd = merge_ds(ds_list, comment_list)

if save_filename != None:
    from pyfusion.data.DA_datamining import DA
    DAtest=DA(dd)
    DAtest.info()
    DAtest.save(save_filename)









