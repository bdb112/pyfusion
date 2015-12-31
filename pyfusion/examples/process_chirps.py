import pyfusion
from pyfusion.acquisition.process_chirps import process_chirps

_var_defaults="""
Ms = None
Ns = [2]
dfdtunit=1
shots = [54185]

# the frequency deviation in kHz allowed before more is considered different
maxd=5

# minimum run length
minlen = 3

plt=True

debug=0
verbose=1
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())


process_chirps(dd, shots=shots, Ns=Ns, Ms=Ms, maxd=maxd, minlen=minlen, plot=plt, debug=debug, verbose=verbose, dfdtunit=dfdtunit) 
