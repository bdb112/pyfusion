# this is the signature of the locked mode in 105396
from pyfusion.convenience import whr, btw

mmm=make_ML_modes(.2,mode_phases=np.array([[-3.1,-2.2,-3.1,-2.7,-2.4]]))
run -i pyfusion/examples/mode_identify_script.py DAfilename='/home/LHD/blackwell/data/datamining/DA/PF2_131122_VSL_6_105_9.npz' sel=range(5) DAfilename=None doN=1 mode_list=mmm
w1 = whr((dd['N']==1) & (freq<0.8))
sh = unique(shot[w1])


for (i,p) in enumerate(decimate(w1,limit=2000)): plot(dd['phases'][p].T,'k', linewidth=0.1, hold=i>0)

for s in sh:
	#  do another select to avoid select of select
        wg=where((shot==s)&(freq<1)&(amp>0.02)&(dd['N']==1))[0]
        if len(wg) > 2:
               sp(dd,t_mid, freq, amp, shot, ind=wg)
               pl.title(str(s))
               pl.figure()
show()
