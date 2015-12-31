""" A VERY small MP+HMP set, just shot 65139 - note that the copy
and the extract() both duplicate data, which is good for development
and debugging, but wasteful of space.
"""
from pyfusion.data.DA_datamining import DA
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.visual.sp import sp
from pyfusion.data.convenience import between, btw, bw, decimate, his, broaden

_var_defaults="""
DAfilename='DA65MP2010HMPno612b5_M_N_fmax.npz'
"""

exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

DA65=DA(DAfilename,load=1)
dd=DA65.copyda() 
DA65.extract(locals())
DA65.info()
print('DA65', DAfilename)

