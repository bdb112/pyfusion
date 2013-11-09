import pylab as pl
import sys

from numpy import intersect1d, pi
from numpy import loadtxt

##
##  See new_mode_identify_script.py
##
# manually enter the mean and sd for three modes, called by color
# n=1
blue = [1.020, 1.488, 1.348, -0.080, 0.989]
bluesd= [0.077, 0.016, 0.048, 0.020, 0.008 ]
# n=0
lblue =[-1.146, -1.094, 0.608, 0.880, 0.164]
lbluesd=[ 0.048, 0.033, 0.034, 0.033, 0.042]
# n=2
red =[2.902, 2.217, 2.823, 1.021, 2.157]
redsd= [0.023, 0.006, 0.028, 0.025, 0.007 ]

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
# This simple strategy works when the number is near zero +- 2Npi,
# which is true for calculating the deviation from the cluster centre.
def twopi(x):
    return ((pi+array(x)) % (2*pi) -pi)


import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())
#run -i ./examples/plot_text_pyfusion.py filename='MP_27233_cf_syama.txt' skip=4
#run -i ./examples/plot_text_pyfusion.py filename=fsfile skip=skip
#sys.argv = ['filename='+fsfile, 'skip='+str(skip)]

oldcolorbar=colorbar
colorbar=0
execfile("./examples/plot_text_pyfusion.py") 
colorbar=oldcolorbar

try:
    oldmodefilename
except:
    oldmodefilename = None

if fsfile != oldmodefilename:
    arr=loadtxt(fsfile,skiprows=skip)
    oldmodefilename = fsfile
else:
    print('reusing old mode file data')
for i,rw in enumerate(arr):
    sh.append(int(rw[0]))
    sd.append(sum(twopi(rw[8:]-blue)**2))
    sdlb.append(sum(twopi(rw[8:]-lblue)**2))
    sdred.append(sum(twopi(rw[8:]-red)**2))
    amp.append(rw[4])
    ind.append(i)

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
pl.suptitle(filename+' mode numbers are wrong')
#for x in array([ds['t_mid'][neq0],1e3*ds['freq'][neq0],neq0]).T.tolist(): text(x[0],x[1],int(x[2])) 
# inds=[5106,5302,5489,1228,1233,1236,478,657,1260] ; average(arr[inds,8:],0); arr[inds,8:]

"""
# now read in HB database into arrays of ne, beta indexed by shot
arr=loadtxt("../../../datamining/LHD_iss_DB07_data_only_selected.csv", skiprows=1,delimiter=',',usecols=[0,1,2,3,4,5,6,7,10,17,18,19])

#test purposes - actuallyneed beta, ne in shot array
neq0 = (array(sdlb) < 1).nonzero()[0]
neq1 = (array(sd) < 1).nonzero()[0]
neq2 = (array(sdred) < 1).nonzero()[0]


beta=arr[:,10]
print ('max beta found is %.2g' % max(beta))

ne=arr[:,7]

shot=arr[:,0]
bet=zeros(60000)
ne=zeros(60000)
shot=arr[:,0]
for i,s in enumerate(shot): 
    bet[s]=arr[i,11]

for i,s in enumerate(shot): 
    ne[s]=arr[i,8]

# We are now ready to plot all neq2 (or 1 etc) shots coded by density or beta
"""
"""
pl.clf()  # gets rid of extra colorbar
rc('font', **{'size':18})
pl.gcf().subplots_adjust(left=0.08, right=1, bottom=0.09, top=0.92)
sind=neq2
#pl.scatter(ds['t_mid'][sind], freq_scale*ds['freq'][sind],dot_size*np.log(ds['amp'][sind]/amp_scale),ds['a12'][sind],hold=0)
pl.scatter(ds['t_mid'][sind], freq_scale*ds['freq'][sind],dot_size*np.log(ds['amp'][sind]/amp_scale),ne[ds['shot'][sind]],hold=0)

colorbar()
pl.gcf().subplots_adjust(left=0.08, right=0.80, bottom=0.09, top=0.92)

pl.ylabel('Frequency (kHz)') ; pl.xlabel('time (S)'); pl.title('N=2 modes, by density HP database DB07_25')

"""

"""
hbshots = [46357,46365,46370,47576,47577,47644,47648,47650,47725]
# These are alternatives for selecting for mode (first example), mode and the high beta subgroup

######################################3
# mode and a hb shot
sind=[]
for sh in hbshots:
    shot_hits = (ds['shot'] == sh).nonzero()[0]
    neq1_hits = (array(sdred) < 1).nonzero()[0]
    hits = intersect1d(shot_hits, neq1_hits)
    print("%d, " % (len(shot_hits))),
    if len(hits)>0: sind.extend(hits)

neq2_highbeta=sind
#neq1_highbeta=sind
#neq0_highbeta=sind

"""
"""
# These appear to be two failed attempts to find fluxstrucs for shots
# in a subrange with particular modes
sind=[]
for sh in shots:
    hits=((ds['shot'] == sh) and (array(sdred) < 1)).nonzero()[0]
    if len(hits)>0: sind.extend(hits)


sind=[]
for sh in shots:
    hits=array((ds['shot'] == sh).tolist() and (array(sdred) < 1).tolist()).nonzero()[0]
    if len(hits)>0: sind.extend(hits)


"""


