# a script file to show Roman Zagórski <roman.zagorski@ifpilm.pl> what the effect of the different profile assumotions is
# 25mm wscl=25 /3.0 ## 40mm – same as below, but use wscl=40, and /2.1
# I used 224_30 as the reference as I obtained the fact values using this shot.  I see that
# when I sent the data to Uwe for the NF paper, I used 0309_17 (see marfe5_20mm.pdf)  why??

# choose one of these
plt.rcParams['figure.figsize']=[10,14]
sys.path.append('/home/bdb112/python')
from copy import deepcopy
wscl = 20; fact = 4.0  
#wscl = 25; fact = 3.0
#wscl = 40; fact = 2.1
ref_shot_DA = 'LP/all_LD/LP20160224_30_L5SEG_2k2.npz'
#ref_shot_DA = 'LP/all_LD/LP20160309_17_L5SEG_2k2.npz'
tit = str('density compensation relative to {f}, assuming 1/e scale of {wscl}mm, factor of {fact}'
          .format(f=ref_shot_DA.split('/')[-1], wscl=wscl, fact=fact))
t_range_list_5marfe=[[0.3,0.32], [0.45,0.47], [0.5,0.52], [.53,0.54], [0.5,0.52]]
dafile_5marfe = 4*['LP/all_LD/LP20160309_52_L5SEG_2k2.npz']; dafile_5marfe.append('LP/all_LD/LP20160309_51_L5SEG_2k2.npz')

#run -i pyfusion/examples/W7X_neTe_profile.py dafile_list=ref_shot_DA labelpoints=0 t_range_list=[0.1,0.2] diag2=ne18 av=np.median 
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list=ref_shot_DA labelpoints=0 t_range_list=[1,1.1] diag2=ne18 av=np.median use_t_mid=1
sigs917=deepcopy(sigs); 
sigsexp=np.array([sigs917[1]*exp(abs(solds[1])/wscl),sigs917[3]*exp(abs(solds[3])/wscl)])/fact

# this suits 224_30
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list=dafile_5marfe labelpoints=0 t_range_list=t_range_list_5marfe diag2=ne18 av=np.nanmean xtoLCFS=0 nrms=sigsexp axset_list="row" ne_lim=[-2,10.5] Te_lim=[-5,60] labelpoints=1 options='1leg,pub'

# this suits 309_17
#run -i pyfusion/examples/W7X_neTe_profile.py dafile_list=dafile_5marfe labelpoints=0 t_range_list=t_range_list_5marfe diag2=ne18 av=np.median xtoLCFS=1 nrms=sigsexp axset_list="row" ne_lim=[-3,18] Te_lim=[-5,60] labelpoints=1 options='1leg,pub'
plt.suptitle(tit)
subplots_adjust(bottom=0.0714, top=0.9581, left=0.0595, right=0.9792, wspace=0.1369, hspace=0.0)
