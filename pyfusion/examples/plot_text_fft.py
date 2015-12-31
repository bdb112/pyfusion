# to reuse old data, use -i option on run (in ipython)
# see the triple commented text for plotting.

import numpy as np
import pylab as pl

_var_defaults="""

phase_array = np.zeros(5)
verbose=1
filename = "g:fft26_50.dat"
xx=4
"""

# this is a way to check existence of variable

try:
    oldfilename
except:
    oldfilename = ""

exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if verbose>1: print(filename, oldfilename)

if oldfilename==filename:
    print('re-using old data - put oldfilename=None to re-read')
else:
    ds = np.loadtxt(fname=filename)

max_shot =  int(np.max(ds[:,0]))
print('max_shot is {m}'.format(m = max_shot))

clock_div=np.zeros(max_shot+1,dtype=int)
last_good = None
times_reused = 0
max_times_reused = 0

for ftcoefs in ds:
    shot = int(ftcoefs[0])
    harms = ftcoefs[array([3,4,6])]
    if max(harms) > 0:
        speedind = np.argsort(harms)[-1]
        if max(ftcoefs[array([1,2])]) < max(harms):
            clock_div[shot] = 2**speedind
            last_good = clock_div[shot]
        else:
            if last_good != None:
                clock_div[shot] = last_good
                times_reused += 1
                max_times_reused = max(max_times_reused,times_reused)


# can use normal savez, but this is much more compact
if np.__version__ > '1.5.0': from numpy import savez 
else: from pyfusion.hacked_numpy_io import savez
savez('a14_clock_div.npz', a14_clock_div=clock_div)

for c in [3,4,5]: 
    pl.semilogy(ds[:,0],.1+ds[:,c],label=str(c))
    pl.legend()
    pl.show()

"""
# plot the components in strips, 10 lots of 100 at t time
chunk = 1000
for start in range(0,100000/chunk):
    dtr = np.transpose(ds[start*chunk:(start+1) * chunk])
    s=dtr[0]

    pl.plot(hold=0)
    for (i,arr) in enumerate(dtr[1:]):
        pl.scatter(s,1+i+0.*s,arr/3)
         
    pl.show()    
    raw_input('CR to continue')
    
"""
