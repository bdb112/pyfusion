from matplotlib import pyplot as plt
from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden
# paste NOt %PASTE
#_PYFUSION_TEST_@@SCRIPT

run -i pyfusion/examples/gen_fs_bands.py  dev_name='W7X' diag_name=W7X_MIRNOV_41_BEST_LOOP shot_range="[[20180912,s] for s in range(43,44)]" max_bands=1 info=0 min_svs=2 max_H=0.999 min_p=0 exception=() outfile='W7X_MIR/preproc/201809/PMIRNOV_41_BEST_LOOP_10_3m_20180912043' fmax=10e3  seg_dt=3e-3 min_svs=2
#  clean up not usually needed
run -i pyfusion/examples/clean_up_pyfusion_text_mp.py MP=0 fileglob="'W7X_MIR/preproc/201809/P*'"
run -i pyfusion/examples/merge_text_pyfusion.py file_list="np.sort(glob('W7X_MIR/preproc/201809/PMIRNOV_41_BEST_LOOP_10_3m_20180912043'))"
run  pyfusion/examples/plot_specgram.py diag_name='W7X_MIR_4136' shot_number=[20180912,43]  NFFT=2048*4
plot_fs_DA(dd);pl.ylim(0,30)
from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file
DA43MIRNOV_13_BEST_LOOP=DA(dd)
DA43MIRNOV_13_BEST_LOOP.save('/tmp/DAMIRNOV_41_13_BEST_LOOP_10_3ms_20180912043.npz")

dd = DA("DAMIRNOV_41_13_BEST_LOOP_10_3ms_20180912043.npz")
w=where((btw(dd['freq'],2,4)) & (dd['amp']>0.012) )[0]
plt.plot(dd['phases'][w].T)
# prettier plots
plt.rcParams['font.size']=20
phs=array([modtwopi(ph,offset=-1) for ph in dd['phases']])
w3=where((btw(dd['freq'],2,4)) & (dd['amp']>0.012) )[0]
plt.plot(phs[w3].T)
plt.figure()
w6=where((btw(dd['freq'],5,7)) & (dd['amp']>0.015) )[0];len(w6)
plt.plot(phs[w6].T)
title('6kHz mode')
ylabel('phase dif n, n-1')
xlabel('probe pairs (n,n-1)')


w=where((btw(dd['freq'],2,4)) & (dd['amp']>0.05) )[0]
pl.plot dd['phases'][w].T

run -i pyfusion/examples/gen_fs_bands.py  dev_name='W7X' diag_name=W7X_MIRNOV_41_BEST_LOOP shot_range="[[20180912,s] for s in range(43,44)]" max_bands=0 seg_dt=1e-3 info=0 min_svs=2 max_H=0.999 min_p=0 exception=() outfile='W7X_MIR/preproc/201809/PMIRNOV_41_BEST_LOOP_20180912043'
time run -i pyfusion/examples/clean_up_pyfusion_text_mp.py MP=0 fileglob="'W7X_MIR/preproc/201809/P*'"
time run pyfusion/examples/merge_text_pyfusion.py file_list="np.sort(glob('W7X_MIR/preproc/201809/P*')" exception=Exception





"""

time run -i pyfusion/examples/clean_up_pyfusion_text_mp.py MP=0 fileglob='/data/datamining/hj/preproc/131128/*2'
time run pyfusion/examples/merge_text_pyfusion.py file_list="np.sort(glob('/data/datamining/hj/preproc/131128/P*2'))" exception=Exception
from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file
DA131128_50_52B=DA(dd)
DA131128_50_52B.save('DA131128_50_52B')
DA131128_50_52B.extract(locals())
w=arange(len(shot))
from pyfusion.visual.sp import sp
from pyfusion.visual.sp import sp
for ii in decimate(np.where((freq<.4) & (amp>0.06)& (a12>0))[0],limit=500): pl.plot(modtwopi(phases[ii],0),'k',linewidth=0.05)
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden
from pyfusion.utils import modtwopi
for ii in decimate(np.where((freq<.4) & (amp>0.06)& (a12>0))[0],limit=500): pl.plot(modtwopi(phases[ii],0),'k',linewidth=0.05)
import pylab as pl
for ii in decimate(np.where((freq<.4) & (amp>0.06)& (a12>0))[0],limit=500): pl.plot(modtwopi(phases[ii],0),'k',linewidth=0.05)
"""
