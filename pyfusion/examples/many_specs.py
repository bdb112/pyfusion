import MDSplus as MDS
#import pyfusion as pf
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.mlab as mlab
from matplotlib.font_manager import FontProperties
import os
from subplots import subplots

from warnings import warn
#import HMA_funcs

shot_list = range(73687, 73736+1)
shot_list = range(73840, 73863)
shot_list = range(73950, 74000)
shot_list = range(73950, 75541)
shot_list = range(74211, 75541)
shot_list = range(75135, 75541)
shot_list = range(71050, 71059)


#desc = dict(shot_list = range(75859,75879+1), comment = 'quick kh scan BPP probe', brief='.05_BPP_2.1V')
desc = dict(shot_list = range(75859,75879+1), comment = 'quick kh scan BPP probe', brief='.05_BPP_2.1V')
shot_list = range(75706, 75792+1) # 4650 scan
desc = dict(shot_list = range(75471, 75513+1)+range(75415,75469+1), comment = '0.01 kh scan 6500A', brief='6500_0.01_2.1V')
#desc = dict(shot_list = range(75706, 75792+1), comment = '0.01 kh scan 4650A', brief='4650_0.01_2.1V')
#7430A shots taken over two days.  Nitrogen dominated - and the earlier ones are probably too.
desc = dict(shot_list = range(74845, 74865+1)+range(74908,74927+1), comment = '0.02 kh scan 7430A', brief='7430_0.02_par')
#desc = dict(shot_list = range(74813, 74835+1), comment = '0.04 kh scan 6950A', brief='6950_0.04_par')
#desc = dict(shot_list = range(74757, 74798+1), comment = '0.04 kh scan 6500A', brief='6500_0.04_par')
#desc = dict(shot_list = range(74720, 74745+1), comment = '0.04 kh scan 4730A', brief='4730_0.04_par')
#desc = dict(shot_list = range(74678, 74704+1), comment = '0.04 kh scan 5100A', brief='5100_0.04_par')
#desc = dict(shot_list = range(74644, 74671+1), comment = '0.04 kh scan 5500A', brief='5500_0.04_par')
#desc = dict(shot_list  = [76474, 76203], comment = 'island scan 6200A', brief='6200A islands', tree='fluctuations')
desc = dict(shot_list  = range(75780, 75896), comment = 'low power 7MHz', brief='6500_lowpower', tree='mirnov')
desc = dict(shot_list  = range(76307, 76345), comment = '22+22kW 7MHz', brief='6500_44kW', tree='mirnov')
desc = dict(shot_list  = range(76274, 76302), comment = '22+22kW 7MHz fine', brief='6500_44kW_fine', tree='mirnov')
desc = dict(shot_list  = range(73641, 73736), comment = '7MHz .02 80kW parallel 7/18 most fail', brief='6500_.02', tree='mirnov')
desc = dict(shot_list  = range(76290, 76291), comment = '7MHz 44kW paper', brief='IAEA', tree='mirnov')
desc = dict(shot_list  = range(79300, 79350), comment = 'Padua 1 4000', brief='Padua', tree='mirnov')

# looks like 76600-699 not on toshiba drive - but maybe not analysed by Shaun
desc=dict(shot_list = range(76615,76662), comment='15 Nov 7MHz', brief='15Nov7', tree='mirnov')
desc=dict(shot_list = range(76739,76870), comment='16 Nov 5MHz', brief='16Nov5', tree='mirnov')
desc=dict(shot_list = range(76870,76893), comment='16 Nov 7MHz', brief='16Nov7', tree='mirnov')
desc=dict(shot_list = range(78474,78540), comment='Aug15 5MHz sweeping', brief='Aug15_5_sweep', tree='mirnov')
#! desc=dict(shot_list = range(76274,76345), comment='Sep27_7MHz', brief='Sep27_7MHz', tree='mirnov')



from bdb_utils import process_cmd_line_args

fft_length = 10242
NFFT_ne =  256*8
#channel = 'mirnov.ACQ132_8:input_02'
channel = 'ACQ132_8:input_03'
tree = 'h1data'
fig = pt.figure(); ax = []
base_dir = 'tmp_pics/'
#base_dir = None
#shot_list = range(75050, 75074)
include_flucstrucs = 0
include_amplifier_sig = 0
include_ne = 1
include_sqrt_ne = 1
include_ne_spec = 1
exception = None
clims = [-100,-10]
mirnov = 'ACQ132_8:input_32'
exec(process_cmd_line_args())

try:
    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())
except:
    warn('Importing process_cmd_line_args allows variables to be changed from the command line')

# extract description:
for k in desc.keys(): exec("{0}=desc['{0}']".format(k))

sweep = 'sweep' in comment

if base_dir == 'None': base_dir = None   # process_cmd confuses strings

if base_dir != None:
    base_dir = base_dir + brief
    try:
        linkdir = base_dir+"/kh_sorted"
        try:
            os.mkdir(base_dir)
        except:
            print('{0} exists I hope'.format(base_dir))
        os.mkdir(linkdir)
    except:
        print("can't make directory {0}".format(linkdir))

for shot in shot_list:
    print shot,
    try:
        #tree_name = MDS.Tree(tree, shot)
        doing = 'mirnov tree'+str(shot)
        tree_name = MDS.Tree('mirnov', shot)
        node = tree_name.getNode(channel)
        data_raw = node.record.data()
        time_raw = node.dim_of().data()
        doing = 'h1data tree'+str(shot)
        h1tree = MDS.Tree('h1data', shot)
        main_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_main:I2').data()
        sec_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_sec:I2').data()
        kh=float(sec_current)/float(main_current)
        if sweep:
            sec_current_end = h1tree.getNode('.operations.magnetsupply.lcu.setup_sec:I3').data()
            kht = kh
            khf = float(sec_current_end)/float(main_current)
            if khf>kht: khf,kht = kht,khf
            khstr = str('$\kappa_H$ {khf:.2f}\dot\dot{kht:.2f}'
                            .format(khf=khf,kht=kht))
            kh = (khf+kht)/2

        else:
            khstr = '$\kappa_H$ {kh:.2f}'.format(kh=kh)

        rf_drive=h1tree.getNode('.RF:rf_drive').data()
        if include_amplifier_sig:
            current_node = tree_name.getNode(mirnov)
            current_raw = current_node.record.data()

        nr = 1+include_ne+include_ne_spec+include_amplifier_sig
        
        (fig, ax) = subplots(nrows = nr, sharex='all', squeeze=False, 
                             apportion=[2.4,1,1])#nrows = 2, sharex = 1)

        rw = 0
        """
        rw = 1; nr += 1  # leave two spots for the specgram
        pos = msp.get_position().get_points().flatten()
        print(pos)
        msp.set_position([pos[0],pos[1]*2./nr,pos[2],pos[3]*nr/(nr-1.)])
        """
        msp = ax[rw,0]

        if include_amplifier_sig: 
            rw += 1
            asp = ax[rw, 0]

        if include_ne: 
            rw += 1
            nep = ax[rw, 0]

        if include_ne_spec: 
            rw += 1
            nesp = ax[rw, 0]
            #nesp.set_position()         


        clr_fig = msp.specgram(data_raw.flatten(), NFFT=fft_length, Fs=1e-3/(time_raw[1]-time_raw[0]), window=mlab.window_hanning, noverlap=int(fft_length*15./16.),cmap='jet',xextent=[np.min(time_raw),np.max(time_raw)])

        if include_amplifier_sig:
            clr_fig2 = asp.specgram(current_raw.flatten(), NFFT=fft_length, Fs=2e6, window=mlab.window_hanning, noverlap=int(fft_length*15./16.),cmap='jet',xextent=[np.min(time_raw),np.max(time_raw)])
        clr_fig[3].set_clim([-100,20])
        clr_fig[3].set_clim(clims)
        
        if include_flucstrucs:
            serial_number = 0; 
            #fs_dictionary, serial_number, success = HMA_funcs.single_shot_fluc_strucs(shot, 'H1ToroidalAxial', [0,0.045], 2048, power_cutoff = 0.05, n_svs = 2)#fft_length)
            time_list = []; freq_list = []
            for serial in fs_dictionary.keys():
                time_list.append(fs_dictionary[serial]['time']/1000)
                freq_list.append(fs_dictionary[serial]['freq'])
            msp.plot(time_list, freq_list, 'k,')

        msp.set_title(r'shot {sh}: {im:.0f}A, {khstr}, 2x{rf:.0f}kW'
                      .format(sh=shot, im=float(main_current), khstr=khstr,
                              rf=200*(np.average(rf_drive/5.))**2))
        msp.set_xlim([np.min(time_raw),np.max(time_raw)])
        msp.set_ylim([0, 125])
        msp.set_xlabel('time (s)')
        msp.set_ylabel('Frequency (kHz)')

        if include_ne or include_ne_spec:
            nenode = h1tree.getNode('.electr_dens:ne_het:ne_centre')
            ne = nenode.data()
            ne_time = nenode.dim_of().data()
                     
        if include_ne or include_ne_spec:
            nep.plot(nenode.dim_of().data(), ne, label='ne(0)')
            try:
                ne_edge = h1tree.getNode('.electr_dens:ne_het:ne_7')
                nep.plot(ne_edge.dim_of().data(),ne_edge.data(),label='ne(r~0.6a)')
            except:
                pass

            if sweep:
                try:
                    ihel = h1tree.getNode('.operations:i_hel')
                    k_h_signal = 1000*((2.17*ihel.data()-0.070)
                                       /float(main_current))
                    nep.plot(ihel.dim_of().data(), k_h_signal, label='k_h')
                except:
                    pass
                

            leg = nep.legend( prop=FontProperties(size='small'))
            leg.get_frame().set_alpha(0.5)

            nep.set_ylim([0, 1.5])
            nep.set_ylabel(r'$n_e$')


        if include_ne_spec:
            doing='ne_spec'
            nesp.specgram(ne.flatten(), NFFT=NFFT_ne, Fs=1e-3/(ne_time[1]-ne_time[0]), window=mlab.window_hanning, noverlap=int(NFFT_ne*15./16.),cmap='jet',xextent=[np.min(ne_time),np.max(ne_time)])
            nesp.set_ylim([0, 50])
            nesp.set_xlim([np.min(time_raw),np.max(time_raw)]) # use the mirnov - it is shorter
            nesp.set_ylabel('density flucts')
            nesp.set_xlabel('time (s)')
        
            if include_sqrt_ne:
                doing='sqrt_ne'
                # why just 290 - maybe the others are not reliable
                from signal_processing import smooth_n
                from node_explorer import Node
                #ne_root = Node(treename='electr_dens',shot=76290)
                ne_root = Node(treename='electr_dens',shot=shot)
                ne=ne_root.NE_HET.NE_7
                t,x = smooth_n(ne.node.data(),901,
                               timebase=ne.node.dim_of().data())
                t_0,x_0 = smooth_n(ne_root.NE_HET.NE_1.node.data(),901,
                               timebase=ne.node.dim_of().data())
                nesp.plot(t,4.5/np.sqrt(x),'w',linestyle='--',linewidth=1.5)
                nesp.set_ylim(0,50)

                msp.plot(t,4.5/np.sqrt(x),'gray',linestyle='--',linewidth=1.5)
                msp.plot(t_0,4.5/np.sqrt(x_0),'w',linestyle='--',linewidth=1.5)
                msp.plot(t_0,23/np.sqrt(x_0),'b',linestyle='--',linewidth=1.5)

        if base_dir != None:
            file_name = '/%d.png'%(shot)
            print ' ', base_dir+file_name, 
            fig.savefig(base_dir+file_name)
            kh_name = "{kh:.3f}_{shot}_{im:.0f}.png".format(kh=kh, shot=shot, im=float(main_current))
            kh_name = linkdir+'/'+kh_name
            os.symlink(".."+file_name, kh_name)
            fig.clf()
            pt.close('all')
        else: 
            thismanager = pt.get_current_fig_manager()
            thismanager.window.wm_geometry("+500+0")
            pt.show()

        #fig.canvas.draw()
        #ax[0].cla()
        print 'success'
    except exception, reason:
        print('failed', doing, reason)

if base_dir == None:
    # need to fix to allow not quitting.
    key = 'y'
    try:
        key=input('return to close all, s to pause')
    except:
        pass
    finally:
        pt.close('all')
 

