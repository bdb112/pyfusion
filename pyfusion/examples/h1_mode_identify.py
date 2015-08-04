# The Initial paste
%run pyfusion/examples/cluster_info.py
figure();co.plot_clusters_phase_lines()
figure();co.plot_kh_freq_all_clusters();ylim(0,80)
from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file
DA_7677=DA('DA_7677.npz')
DA_7677.extract(locals())
 
sht=76670 #86514 # 76670   # 672, 616

# typically paste from here down 
thr = 1.5
scl=1500
clim=None
figure()
boydsdata=1
if boydsdata:
    ph = -phases
else:
    ph = phases

w15=where((ml[15].one_rms(ph)<thr) & (shot==sht))[0];len(w15)
w4=where((ml[4].one_rms(ph)<thr) & (shot==sht))[0];len(w4)
w=union1d(w4,w15)
if len(w)>0: scatter(t_mid[w],freq[w],scl*amp[w],c='b',label='n=5/m=4')

w6=where((ml[6].one_rms(ph)<thr) & (shot==sht))[0];len(w6)
w1=where((ml[1].one_rms(ph)<thr) & (shot==sht))[0];len(w1)
w10=where((ml[10].one_rms(ph)<thr) & (shot==sht))[0];len(w10)
w=union1d(w1,w10)
if len(w)>0: scatter(t_mid[w],freq[w],scl*amp[w],c='g',label='n=4/m=3')
legend()
%run -i pyfusion/examples/plot_specgram.py shot_number=sht dev_name=H1Local "diag_name='H1ToroidalAxial'" NFFT=2048 noverlap=NFFT*7/8 hold=1
ylim(0,100)
if clim is not None:
    pl.clim(clim)

"""

#font={'family' : 'arial', 'weight' : 'bold', 'size' : 16};matplotlib.rc('font',**font) # no arial on t440p
font={'size' : 16};matplotlib.rc('font',**font)
scl=3000
"""
