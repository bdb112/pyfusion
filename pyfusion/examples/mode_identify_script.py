"""
This version is a memory hog - 9X , 5 modes took 90gB (14 Nov 2013)

w=where(array(sd)<10)[0]
for ii in decimate(w,limit=2000): pl.plot(dd["phases"][ii],'k',linewidth=0.02)
mode.plot()

dd['N']=-1+0*dd["shot"].copy()

run -i pyfusion/examples/cluster_DA.py DA_file='PF2_120229_MP_27233_27233_1_256.npz'
ml = co.make_mode_list(min_kappa=3.5)

from pyfusion.data.DA_datamining import DA, report_mem
DA233=DA(DA_file,load=1)
dd=DA233.da.copy()  # use copy here to allow re-runs 
run -i pyfusion/examples/mode_identify_script.py doN=1 sel=arange(0,6) threshold=1 mode_list=ml
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden
his NN

"""
import pylab as pl
import sys

from numpy import intersect1d, pi
from numpy import loadtxt
import numpy as np
from pyfusion.clustering.modes import Mode

# This simple strategy works when the number is near zero +- 2Npi,
# which is true for calculating the deviation from the cluster centre.
# does not attempt to make jumps small (might be a good idea!)
def twopi(x, offset=pi):
    return ((offset+np.array(x)) % (2*pi) -offset)


shot_list = []


# manually enter the mean and sd for the modes, called by color
# n=1
OM=[]  # old modes (pre MP2010)
OM.append(Mode('N=1', N=1, NN=100, cc = [1.020, 1.488, 1.348, -0.080, 0.989], csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
#blue.csd= [0.077, 0.016, 0.048, 0.020, 0.008 ]


OM.append(Mode('N=1,46747', N=1, NN=101, cc = [0.6, 1.7, 0.8, 0.2, 0.989], csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
# obscured by other noise....


OM.append(Mode('N=1,or 0?', N=1, NN=102, cc = [1.0, -0.65, 1.84, 1.8, 1.95], csd= [0.2, 0.2, 0.3, 0.20, 0.2 ]))
# very clear on 38100


OM.append(Mode('N=1 - broad', N=1, NN=103, cc =[0.3, 0.25, 1.3, 2.0, 1.8], csd=[ 0.48, 0.5, 0.5, 0.5, 0.42]))

OM.append(Mode('N=1 - residual', N=1, NN=104, cc =[1.2, 0.86, 1.8, -0.1, 1.0], csd=[ 0.2, 0.2, 0.3, 0.3, 0.3]))


# n=0
OM.append(Mode('N=0', N=0, NN=50, cc =[-1.146, -1.094, 0.608, 0.880, 0.164], csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))
# mainly low 27s
#lblue.csd=[ 0.048, 0.033, 0.034, 0.033, 0.042]


OM.append(Mode('N=0 - fract', N=0, NN=51, cc =[0.3, -.75, 0.608, 0.880, 0.45], csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))

OM.append(Mode('N=0 - fract1', N=0, NN=52, cc =[-2.1, -1.1, 0.5, 1.06, 0.17], csd=[ 0.3, 0.33, 0.34, 0.33, 0.25]))



# n=2
OM.append(Mode('N=2', N=2, NN=200, cc =[2.902, 2.217, 2.823, 1.021, 2.157], csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
#red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
# saved as 200  with sumsd2 <10


OM.append(Mode('N=2, LF', N=2, NN=201, cc =[2., 2.4, 2.0, 1.5, 2.2], csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
#red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
# saved 201 with sumsd2 <10 


OM.append(Mode('N=2, 60k', N=2, NN=202, cc =[2.3, 2.4, 1.7, 2.1, 1.7], csd= [0.3, 0.2, 0.2, 0.2, 0.2 ]))
#red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
# saved NN=202 sumsd2<10  - -model is 47634 (3928 overlap with  redlow)

OM.append(Mode('N=2, 60k MP1', N=2, NN=204, cc =[0, 2.4, 1.7, 2.1, 1.7], csd= [5, 0.2, 0.2, 0.2, 0.2 ],shot_list=[54184,54185,54194,54195,54196,54197,54198]))
# ref60, but allow for MP1 to be dead (large csd)

OM.append(Mode('W1', N=2, NN=203, cc =[2.5, 0.3, -1.4, 0.7, 2.2], csd =[0.5, 0.5, 0.3, 0.3, .25]))
# too rare to worry

def f1(cc):
    return( -twopi(np.array(cc) + np.pi, offset=4))
    
MP2010=[]  # old modes (pre MP2010)
MP2010.append(Mode('N=2', N=2, NN=200, cc = f1([1.020, 1.488, 1.348, -0.080, 0.989]), csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
#blue.csd= [0.077, 0.016, 0.048, 0.020, 0.008 ]


MP2010.append(Mode('N=2,46747', N=2, NN=201, cc = f1([0.6, 1.7, 0.8, 0.2, 0.989]), csd= [0.4, 0.2, 0.3, 0.20, 0.2 ]))
# obscured by other noise....


MP2010.append(Mode('N=2,or 0?', N=2, NN=202, cc = f1([1.0, -0.65, 1.84, 1.8, 1.95]), csd= [0.2, 0.2, 0.3, 0.20, 0.2 ]))
# very clear on 38100


MP2010.append(Mode('N=2 - broad', N=2, NN=203, cc =f1([0.3, 0.25, 1.3, 2.0, 1.8]), csd=[ 0.48, 0.5, 0.5, 0.5, 0.42]))

MP2010.append(Mode('N=2 - residual', N=2, NN=204, cc =f1([1.2, 0.86, 1.8, -0.1, 1.0]), csd=[ 0.2, 0.2, 0.3, 0.3, 0.3]))


# n=0
MP2010.append(Mode('N=3', N=3, NN=300, cc =f1([-1.146, -1.094, 0.608, 0.880, 0.164]), csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))
# mainly low 27s
#lblue.csd=[ 0.048, 0.033, 0.034, 0.033, 0.042]


MP2010.append(Mode('N=3 - fract', N=3, NN=301, cc =f1([0.3, -.75, 0.608, 0.880, 0.45]), csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))

MP2010.append(Mode('N=-3  ', N=-3, NN=-301, cc =([-2.74, +2.1, -2.7, -2.3, -2.5]), csd=[ 0.48, 0.33, 0.34, 0.33, 0.42]))

MP2010.append(Mode('N=3 - fract1', N=3, NN=302, cc =f1([-2.1, -1.1, 0.5, 1.06, 0.17]), csd=[ 0.3, 0.33, 0.34, 0.33, 0.25]))

# was n=2
MP2010.append(Mode('N=1', N=1, NN=100, cc =f1([2.902, 2.217, 2.823, 1.021, 2.157]), csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
#red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
# saved as 200  with sumsd2 <10


MP2010.append(Mode('N=1, LF', N=1, NN=101, cc =f1([2., 2.4, 2.0, 1.5, 2.2]), csd= [0.2, 0.2, 0.2, 0.3, 0.2 ]))
#red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
# saved 201 with sumsd2 <10 


MP2010.append(Mode('N=1, 60k', N=1, NN=102, cc =f1([2.3, 2.4, 1.7, 2.1, 1.7]), csd= [0.3, 0.2, 0.2, 0.2, 0.2 ]))
#red.csd= [0.023, 0.006, 0.028, 0.025, 0.007 ]
# saved NN=202 sumsd2<10  - -model is 47634 (3928 overlap with  redlow)

MP2010.append(Mode('N=1, 60k MP1', N=1, NN=104, cc =f1([0, 2.4, 1.7, 2.1, 1.7]), csd= [5, 0.2, 0.2, 0.2, 0.2 ],shot_list=[54184,54185,54194,54195,54196,54197,54198]))
# ref60, but allow for MP1 to be dead (large csd)

MP2010.append(Mode('W1?', N=1, NN=103, cc =f1([2.5, 0.3, -1.4, 0.7, 2.2]), csd =[0.5, 0.5, 0.3, 0.3, .25]))
# too rare to worry

# need to track down with MP1 should be with another shot
MP2010.append(Mode('N=-1',N=-1, NN=-100, cc=[-0.416, -0.814, -1.386, -1.190, -1.147], csd = [4, 0.060, 0.041, 0.086, 0.072]))
# from the N~0 cases where M was found but not M, in 65139
MP2010.append(Mode('N~0',N=0, NN=0, cc=[-0.829, -0.068, -0.120, 0.140, -0.032],csd=  [3, 0.025, 0.044, 0.037, 0.029]))
# and this from the other ones (N ~ -1)
MP2010.append(Mode('N=-1',N=-1, NN=-101, cc=[-0.454, -0.775, -1.348, -1.172, -1.221],  csd=[ 3, 0.071, 0.048, 0.133, 0.113]))

ideal_modes=[]
ideal = np.load('ideal_toroidal_modes.npz')['subset']
ideal = ideal[:,arange(5)]  # need to adjust
for i in range(10):
    N=i-5
    ideal_modes.append(Mode('N={N}'.format(N=N),N=N, NN=i*(100), cc=ideal[i], csd=0.5*np.ones(len(ideal[0]))))

ind = None
mode_list = MP2010
mode_list = ideal_modes
mode=None
threshold=None
mask=None
doM = False
doN = False
sel = arange(11,16)

import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())

if not hasattr(dd,'has_key'):
    raise LookupError("dd not loaded into memory - can't store")

if mode==None: mode = mode_list[0]
if not(doM) and not(doN): raise ValueError('Need to choose doN=True and/or doM=True')

if ind == None: ind = arange(len(dd['shot']))
# the form phases = dd['phases'][ind,11:16] consumes less memory
if (sel is not None) and  (np.average(np.diff(sel))==1):   # smarter version
    phases = dd['phases'][ind,sel[0]:sel[-1]+1]
else:
    phases = dd["phases"][ind]
    if sel is not None:
        phases = phases.T[sel].T
#phases = np.array(phases.tolist())

sd = mode.std(phases)

#  generate mode number entries if not already there.
for mname in 'N,NN,M,MM,mode_id'.split(','):
    if not(mname in dd.keys()):
        use_dtype=int16
        minint = np.iinfo(use_dtype).min
        dd[mname] = minint*np.ones(len(dd['shot']),dtype=use_dtype)

# Note: if you re-run the script, num_set in the modes will only reset for
# modes defined above
tot_set, tot_reset = (0,0)


for mode in mode_list:
    # careful - better to use keywords than positional args
    if doN: mode.store(dd, threshold=threshold, mask=mask, sel=sel)
    if doM: mode.storeM(dd, threshold=threshold, sel=sel)

    tot_set   += mode.num_set
    tot_reset += mode.num_reset
print('Total set = {t}, reset = {r}'.format(t=tot_set, r=tot_reset))

"""
29th Nov method- obsolete

sd = []
sdlb = []
sdred = []

sh = []
amp = []
ind = []
#arr=loadtxt('MP_27233_cf_syama.txt',skiprows=4)
# arr=loadtxt('MP512all.txt',skiprows=4)
# read in the delta phases from the aggregated pyfusion database
# and build up a list of flucstrucs and the sum of squares relative to the
# three modes.
#fsfile='MP512all.txt'
#fsfile='PF2_120229_MP_27233_27233_1_256.dat'

## let plot_text_pyfusion choose the file and skip
#fsfile='PF2_120229_MP_50633_50633_1_256.dat'
#skip=4
hold=1
dt=140e-6
ind = None
mode=redlow

import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())
#run -i ./examples/plot_text_pyfusion.py filename='MP_27233_cf_syama.txt' skip=4
#run -i ./examples/plot_text_pyfusion.py filename=fsfile skip=skip
#sys.argv = ['filename='+fsfile, 'skip='+str(skip)]

#execfile("./examples/plot_text_pyfusion.py") 

if ind == None: ind = arange(len(dd['shot']))
phases = dd["phases"][ind]

for i in ind:
    if (i % 100000) == 0: print(i),
    sd.append(sum((twopi(dd["phases"][i]-mode.cc)/mode.csd)**2))
    sh.append(dd["shot"][i])
"""
"""
exec(pyfusion.utils.process_cmd_line_args())

# find the indices of modes within a  short distance from the classification
neq0 = (array(sdlb) < 1).nonzero()[0]
neq1 = (array(sd) < 1).nonzero()[0]
neq2 = (array(sdred) < 1).nonzero()[0]

# do a crude plout by color=mode only
msize=40
print(hold)
pl.scatter(dt+ds['t_mid'][neq0],fsc*ds['freq'][neq0],hold=hold,label='N=0',c='cyan',s=msize)
pl.scatter(dt+ds['t_mid'][neq1],fsc*ds['freq'][neq1],label='N=1',s=msize)
pl.scatter(dt+ds['t_mid'][neq2],fsc*ds['freq'][neq2],label='N=2',c='red',s=msize)

pl.legend()
pl.suptitle(filename)
#for x in array([ds['t_mid'][neq0],1e3*ds['freq'][neq0],neq0]).T.tolist(): text(x[0],x[1],int(x[2])) 
# inds=[5106,5302,5489,1228,1233,1236,478,657,1260] ; average(arr[inds,8:],0); arr[inds,8:]

"""
