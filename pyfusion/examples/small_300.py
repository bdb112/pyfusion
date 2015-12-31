""" A small MP+HMP set, just shot 65139 - note that the copy
and the extract() both duplicate data, which is good for development
and debugging, but wasteful of space.
"""
from pyfusion.data.DA_datamining import DA
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.visual.sp import sp
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden

_var_defaults="""
DAfilename='300_small.npz'
"""
exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

DA300=DA(DAfilename,load=1)
dd=DA300.copyda() 
DA300.extract(locals())
DA300.info()
print('DA300', DAfilename)
