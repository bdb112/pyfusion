# must be manually pasted (not %paste)
import pyfusion
import numpy as np
import matplotlib.pyplot as plt
import json
import bz2 

"""
_PYFUSION_TEST_@@block=0
shot=[20160217,16] # [20160309,3]
pyfusion.config.set('global','localdatapath','/data/datamining/local_data/W7X/bad')
run -i pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name='W7X_L53_LP01_I' shot_number=shot
pyfusion.config.set('global','localdatapath','/data/datamining/local_data/W7X/0309')
cdata=data
run -i pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name='W7X_L53_LP01_I' shot_number=shot
plot cdata.timebase,cdata.signal/3277.24,'--r'
ylim(-32.33,-32.31);xlim(.5000018,.5000021)  # for 309_3_L53 LP01_U
ylim(-.005,.005);xlim(.599,.602) #  for 309_3_L53_LP01_I and LP02_I


s=array(sig); w = where(abs(f*s[1:]-s[0:-1])<abs(s[1:]-s[0:-1]))[0]; len(w)
s=array(sig); w = where(abs(s[1:]-f*s[0:-1])<abs(s[1:]-s[0:-1]))[0]; len(w)
# works except when the point before the transition is 0


2016_0224_20_L53_LP02_I_signal.json@from=1456316272989099901&upto=1456316339989099900 .85 secs
%time xx=json.load(bz2.BZ2File('/data/datamining/local_data/W7X/json/2016_0224_20_L53_LP02_I_signal.json@from=1456316272989099901&upto=1456316339989099900.bz2'))
1;45 sec (file is about 16x smaller)
"""
block=0
#ds = json.load(bz2.BZ2File('/data/datamining/local_data/W7X/json/2016_0224_20_L53_LP02_I_signal.json@from=1456316272989099901&upto=1456316339989099900.bz2'))
ds = json.load(bz2.BZ2File('/data/datamining/local_data/W7X/json/2016_0224_18_L53_LP02I_signal.json@from=1456314322611207561&upto=1456314322709259561.bz2'))
signal = np.array(ds['values'])
tb = np.array(ds['dimensions'])
fact = 3277.24
# the 0 values are unchanged by the factor - so remove these for now
wnz = np.where(signal != 0)[0]
wz = np.where(signal == 0)[0]
sig = signal[wnz]
wstart = np.where(np.abs(sig[1:]-fact*sig[0:-1]) < np.abs(sig[1:]-sig[0:-1]))[0]
wend = np.where(np.abs(fact*sig[1:]-sig[0:-1]) < np.abs(sig[1:]-sig[0:-1]))[0]
if len(wstart) != len(wend):
    raise LookupError('Warning unequal starts and ends')

elif len(wstart) > 0:
    print('intervals lengths are {ints}'.format(ints=wend-wstart))
    for st, en in zip(wstart, wend)[0:10]:
        print('{st:10,} - {en:10,}'.format(st=st, en=en))

    for st, en in zip(wstart, wend):
            inds = np.arange(st+1, en+1)
            sig[inds] = sig[inds]/fact

# now have to scatter the data back into the nonzero locations - the zero locations are already there.
corrected_signal = signal.copy()
corrected_signal[wnz] = sig


plt.plot((tb-tb[0])/2000 - np.arange(0,len(tb)),',')
plt.ylim(-2000, 2000)
plt.ylabel('timebase offset in samples')
plt.show(block)

