""" Script file to help with W7X_neTe_profile for fine tuning etc, especially now we include MPM
""" 
da_file = "LP/to_daihong/LP20160309_7_L5SEG_2k2am1p2_21_0_1s.npz" ; dk = 18.0; max_ne=4.6
#da_file = "LP/all_LD/LP20160309_17_L5SEG_2k2.npz" ; dk = 18.0; max_ne=4.6
# da_file='LP/LP20160309_10_L5SEG_amoeba21_1.2_2k.npz'
# da_file='LP/to_daihong/LP20160309_32_L57__2k2am1p2_21_p046_1p14.npz'; dk = 25.0; max_ne=3.2
# da_file="LP/to_daihong/LP20160309_13_L57__2k2am1p2_21_p165_1p115.npz" ; dk = 22.0; max_ne=3.6
run -i pyfusion/examples/W7X_neTe_profile.py dafile_list="[da_file]" labelpoints=1 t_range_list=[[0.3,0.4]] diag2=ne18 av=np.median xtoLCFS=1  axset_list="row"  ne_lim=[0,8] Te_lim=[0,90]

MPM_key = [None, "SOL", "reff"][xtoLCFS]
if MPM_key is not None:
    try:
        MPM = read_MPM_data('/data/databases/W7X/MPM/MPM_{0:d}_{1:d}.zip'.format(*shot_number),verbose=1)
        dist = 1000 * MPM[MPM_key] - (xtoLCFS == 2) * reff_0  # reff_0 is already in mm
        axLCne.plot(dist, MPM['ne18'],'b', label='MPM')
        axLCne.plot(-dist, MPM['ne18'],'b')
        axLCne.set_ylim(.2,5)
        max_x = np.max([d for d in dist if d != inf])
        axLCne.set_xlim(max_x * np.array([-1,1]))
        axLCTe.plot(dist, MPM['Te'],'b', label='MPM')
        axLCTe.plot(-dist, MPM['Te'],'b')
    except IOError:
        pass
    finally:
        axLCne.set_ylim(.2,5)
    
d_vec=linspace(0,axLCne.get_xlim()[1])
rfact = [1, 1, 0.6][xtoLCFS]
axLCne.plot(d_vec, max_ne*exp(-d_vec*rfact/dk),':g', label=str('exp {0:.1f}mm'.format(dk/rfact)))
axLCne.plot(-d_vec, max_ne*exp(-d_vec*rfact/dk),':g')
axLCne.plot([0,0], axLCne.get_ylim(),'k',lw=0.3)
axLCne.legend(prop=dict(size='medium'), loc='best')
figLCFS.subplots_adjust(bottom=0.1, top=0.9, left=0.0804, right=0.9702, wspace=0.2, hspace=0.0)
