# this is the signature of the locked mode in 105396
# this file must be pasted, directly (%paste objects to the run ... )
# at the moment, need to run with mode_list=None to compile make mode
# very convenient to run again on another DA - recall the run command,
# and %paste the commands after.
# Also, can delete any uninteresting shots and save the remainder as pngs 
# from pyfusion.visual import window_manager, sp, tog
# window_manager.save('103')
#_PYFUSION_TEST_@@SCRIPT

from pyfusion.data.convenience import whr, btw

mmm=make_ML_modes(.2,mode_phases=np.array([[-3.1,-2.2,-3.1,-2.7,-2.4]]))
# old version used to require DAfilename=None - not automatic
# /data/datamining/DA/PF2_131122_VSL_6_105_9.npz
#run -i pyfusion/examples/mode_identify_script.py DAfilename='saved_PF2_150311_VSL6_105.npz' sel=range(5) DAfilename=None doN=1 mode_list=mmm
run -i  pyfusion/examples/mode_identify_script.py DAfilename='/data/datamining/DA/saved_PF2_150311_VSL6_105.npz' sel=range(5)  doN=1 mode_list=mmm

# from here on can be %paste ed
thisDA.extract(locals())
N=dd['N']
w1 = whr((dd['N']==1) & (freq<0.8))
sh = unique(shot[w1]) ; len(sh)

if pl.gcf().get_label()!='':
        pl.figure()   # avoid overwriting labeled figures

for (i,p) in enumerate(decimate(w1,limit=2000)): plot(dd['phases'][p].T,'k', linewidth=0.1, hold=i>0)

from pyfusion.visual import window_manager, sp, tog

for s in sh:
	#  do another select to avoid select of select
        wg_loose = where((shot==s)&(freq<2)&(amp>0.015)&(dd['N']==1))[0]
        if len(where(freq[wg_loose]<1)[0]) > 2:  # at least 2  below 1kHz
               pl.figure(num=str(s))
               coll = sp(dd,t_mid, freq, amp, N, ind=wg_loose, size_scale=.01) # marker='v')
               ws = where((shot==s))[0]
               ax2 = coll.axes.twinx()
               ax2.plot(t_mid[ws], w_p[ws])  # ,hold=1) only for simple plot
               #sp(dd, t_mid, w_p/200, amp, 1/(1+freq), size_scale=.01,ind=ws,hold=1)
               pl.title(str(s))

show()
