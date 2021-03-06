from __future__ import print_function
from . import clustering as clust
from pyfusion.devices.H1 import H1_scan_list
from multiprocessing import Process, Pool
import pyfusion as pf
#import MDSplus as MDS
import numpy as np
import time, os, copy, itertools

def single_shot_fluc_strucs(shot=None, array=None, other_arrays=None, other_array_labels=None, start_time=0.001, end_time = 0.08, samples=1024, power_cutoff = 0.1, n_svs = 2, overlap = 4, meta_data=None):
    '''This function will extract all the important information from a
    flucstruc and put it into the form that is useful for clustering
    using hte clustering module.

    SH: 8June2013 '''
    print(os.getpid(), shot)
    time_bounds = [start_time, end_time]

    #extract data for array, naked_coil and ne_array, then reduce_time, interpolate etc...
    data = pf.getDevice('H1').acq.getdata(shot, array).reduce_time([start_time, end_time])
    data = data.subtract_mean(copy=False).normalise(method='v',separate=True,copy=False)
    data_segmented = data.segment(samples,overlap=overlap, datalist = 1)
    print(other_arrays, other_array_labels)
    if other_arrays is None: other_array_labels = []
    if other_arrays is None: other_arrays = []; 
    if meta_data is None : meta_data = []

    #Get the naked coil and interferometer array if required
    #Need to include the standard interferometer channels somehow.
    other_arrays_segmented = []
    for i in other_arrays:
        tmp = pf.getDevice('H1').acq.getdata(shot, i).change_time_base(data.timebase)
        other_arrays_segmented.append(tmp.segment(samples, overlap = overlap, datalist = 1))
    instance_array_list = []
    misc_data_dict = {'RMS':[],'time':[], 'svs':[]}

    #How to deal with the static case?
    for i in other_array_labels: 
        if i[0]!=None:  misc_data_dict[i[0]] = []
        if i[1]!=None:  misc_data_dict[i[1]] = []
    #This should probably be hard coded in... 
    fs_values = ['p','a12','H','freq','E']
    for i in meta_data: misc_data_dict[i]=[]
    for i in fs_values: misc_data_dict[i]=[]

    #Cycle through the time segments looking for flucstrucs
    for seg_loc in range(len(data_segmented)):
        data_seg = data_segmented[seg_loc]
        time_seg_average_time = np.mean([data_seg.timebase[0],data_seg.timebase[-1]])
        fs_set = data_seg.flucstruc()
        #Need to check my usage of rfft.... seems different to scipy.fftpack.rfft approach
        other_arrays_data_fft = []
        for i in other_arrays_segmented:
            other_arrays_data_fft.append(np.fft.rfft(i[seg_loc].signal)/samples)
            if not np.allclose(i[seg_loc].timebase,data_seg.timebase): 
                print("WARNING possible timebase mismatch between other array data and data!!!")
        d = (data_seg.timebase[1] - data_seg.timebase[0])
        val = 1.0/(samples*d)
        N = samples//2 + 1
        frequency_base = np.round((np.arange(0, N, dtype=int)) * val,4)
        #get the valid flucstrucs
        valid_fs = []
        for fs in fs_set:
            if (fs.p > power_cutoff) and (len(fs.svs()) >= n_svs): valid_fs.append(fs)
        #extract the useful information from the valid flucstrucs
        for fs in valid_fs:
            for i in fs_values: misc_data_dict[i].append(getattr(fs,i))
            misc_data_dict['svs'].append(len(fs.svs()))
            #for i in meta_values: misc_data_dict[i].append(eval(i))
            for i in meta_data:
                try:
                    misc_data_dict[i].append(copy.deepcopy(data.meta[i]))
                except KeyError:
                    misc_data_dict[i].append(None)
            misc_data_dict['RMS'].append((np.mean(data.scales**2))**0.5)
            misc_data_dict['time'].append(time_seg_average_time)
            #other array data
            tmp_loc = np.argmin(np.abs(misc_data_dict['freq'][-1]-frequency_base))
            for i,tmp_label in zip(other_arrays_data_fft, other_array_labels):
                if tmp_label[0]!=None: misc_data_dict[tmp_label[0]].append(np.abs(i[:,0]))
                if tmp_label[1]!=None: misc_data_dict[tmp_label[1]].append(np.abs(i[:,tmp_loc]))
            phases = np.array([tmp_phase.delta for tmp_phase in fs.dphase])
            phases[np.abs(phases)<0.001]=0
            instance_array_list.append(phases)
    #convert lists to arrays....
    for i in misc_data_dict.keys():misc_data_dict[i]=np.array(misc_data_dict[i])
    return np.array(instance_array_list), misc_data_dict

def single_shot_svd_wrapper(input_data):
    return single_shot_fluc_strucs(*input_data)

def multi_svd(shot_selection,array_name, other_arrays = None, other_array_labels = None, meta_data = None,
    n_cpus=8, NFFT = 2048, power_cutoff=0.05, min_svs=2, overlap = 4,): 
    '''Runs through all the shots in shot_selection other_arrays is a
    list of the other arrays you want to get information from '''

    #Get the scan details 
    shot_list, start_times, end_times = H1_scan_list.return_scan_details(shot_selection) 
    rep = itertools.repeat
    if other_arrays is None: other_arrays = ['ElectronDensity','H1ToroidalNakedCoil']
    if other_array_labels is None: other_array_labels = [['ne_static','ne_mode'],[None,'naked_coil']]
    if meta_data is None : meta_data = ['kh','heating_freq','main_current','sec_current', 'shot']

    input_data_iter = itertools.izip(shot_list, rep(array_name),
                                     rep(other_arrays),
                                     rep(other_array_labels),
                                     start_times, end_times,
                                     rep(NFFT), rep(power_cutoff),
                                     rep(min_svs), rep(overlap),rep(meta_data))
    #generate the shot list for each worker
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print('creating pool map')

        results = pool.map(single_shot_svd_wrapper, input_data_iter)
        print('waiting for pool to close ')
        pool.close()
        print('joining pool')
        pool.join()
        print('pool finished')
    else:
        results = map(single_shot_svd_wrapper, input_data_iter)
    start=1
    for i,tmp in enumerate(results):
        print(i)
        if tmp[0]!=None:
            if start==1:
                instance_array = copy.deepcopy(tmp[0])
                misc_data_dict = copy.deepcopy(tmp[1])
                start = 0
            else:
                instance_array = np.append(instance_array, tmp[0],axis=0)
                for i in misc_data_dict.keys():
                    misc_data_dict[i] = np.append(misc_data_dict[i], tmp[1][i],axis=0)
        else:
            print('One shot may have failed....')
    return clust.feature_object(instance_array = instance_array, misc_data_dict = misc_data_dict)
    

def get_array_data(current_shot, array_name, time_window=None,new_timebase=None):
    array_cutoff_locs = [0]
    data = pf.getDevice('H1').acq.getdata(current_shot, array_name)
    if new_timebase!=None:
        print('interpolating onto a new timebase....')
        data = data.change_time_base(new_timebase)
    if time_window!=None:
        data = data.reduce_time(time_window)
    return data

def find_peaks(data_fft, n_pts=20, lower_freq = 1500,by_average=True, moving_ave=5, peak_cutoff = 0):
    time_pts, n_dims, freqs = data_fft.signal.shape
    good_indices = []
    for i in range(time_pts):
        good_indices_tmp = []
        if by_average:
            tmp = np.average(np.abs(data_fft.signal[i,:,:]**2),axis=0)
        else:
            tmp = np.product(np.abs(data_fft.signal[i,:,:]),axis=0)
        peaks = np.zeros(data_fft.signal.shape[2],dtype=bool)
        how_peaked = np.zeros(data_fft.signal.shape[2],dtype=float)

        #Find the peaks in the data
        if moving_ave!=None:
            moving_peak = tmp*1
            moving_peak[(moving_ave-1)/2:len(moving_peak) - (moving_ave-1)/2] = np.convolve(tmp,np.ones(moving_ave,dtype=float)/moving_ave,'valid')
            how_peaked = tmp - moving_peak
        else:
            how_peaked[1:-1] = np.abs(tmp[0:-2]-tmp[1:-1])+ np.abs(tmp[1:-1]-tmp[2:])

        peaks[1:-1] = (tmp[0:-2]<tmp[1:-1]) * (tmp[1:-1]>tmp[2:])

        #Check how much of a peak the peaks are...
        how_peaked = how_peaked * peaks

        #Sort by 'peakedness'
        tmp_indices = np.argsort(how_peaked)
        if len(tmp_indices)<n_pts:
            n_pts_tmp = len(tmp_indices)
        else:
            n_pts_tmp = n_pts

        #Take the best n_pts_tmp values - exclude low frequency
        for j in tmp_indices[-n_pts_tmp:]:
            if (np.abs(data_fft.frequency_base[j]-0)>lower_freq) and (how_peaked[j]>peak_cutoff):
                good_indices_tmp.append(j)
        good_indices.append(good_indices_tmp)
        if len(np.unique(good_indices_tmp)) != len(good_indices_tmp):print("!!!!!!!!!!!!! duplicate problem")
    return good_indices

def return_values(tmp_array, good_indices, force_index = None):
    for i, tmp_indices in enumerate(good_indices):
        if force_index!=None: tmp_indices=np.array(tmp_indices)*0 + force_index
        if i==0: 
            tmp_data = tmp_array[i,:,tmp_indices]
        else:
            tmp_data = np.append(tmp_data,tmp_array[i,:,tmp_indices],axis=0)
    return tmp_data

def return_non_freq_dependent(tmp_array, good_indices,force_index = None):
    for i, tmp_indices in enumerate(good_indices):
        if force_index!=None: tmp_indices=np.array(tmp_indices)*0 + force_index
        if i==0: 
            tmp_data = tmp_array[tmp_indices]
        else:
            tmp_data = np.append(tmp_data,tmp_array[tmp_indices],axis=0)
    return tmp_data

def return_time_values(tmp_array, good_indices):
    for i, tmp_indices in enumerate(good_indices):
        tmp_indices=np.array(tmp_indices)*0 + i
        if i==0: 
            tmp_data = tmp_array[tmp_indices]
        else:
            tmp_data = np.append(tmp_data,tmp_array[tmp_indices],axis=0)
    return tmp_data

def extract_data_by_picking_peaks(current_shot, array_names,NFFT=1024, hop=256,n_pts=20,lower_freq=1500, ax = None, time_window = [0.004,0.090]):
    #Get Mirnov Data, ne_array data and naked coil data, and put them all on the same timebase
    data = get_array_data(current_shot, array_names[0], time_window = time_window)
    timebase = data.timebase
    data_fft = data.generate_frequency_series(NFFT,hop)

    #if other_arrays is None: other_arrays = []
    #if meta_data is None : meta_data = []
    #if other_arrays is None: other_array_labels = []
    #other_arrays_segmented = []
    #for i in other_arrays:
    #    tmp = pf.getDevice('H1').acq.getdata(shot, i).change_time_base(data.timebase)
    #    other_arrays_segmented.append(tmp.generate_frequency_series(NFFT,hop))

    if current_shot<60000:
        get_ne_array=0; get_naked_coil = 0
    else:
        get_ne_array=0; get_naked_coil = 0

    if get_ne_array:
        ne_data = get_array_data(current_shot, "ElectronDensity",new_timebase = timebase)
        ne_fft = ne_data.generate_frequency_series(NFFT,hop)
    if get_naked_coil:
        naked_coil = get_array_data(current_shot, "H1ToroidalNakedCoil",new_timebase = timebase)
        naked_coil_fft = naked_coil.generate_frequency_series(NFFT,hop)
    #Find the best peaks in the data
    good_indices = find_peaks(data_fft, n_pts=n_pts, lower_freq = lower_freq)

    #Use the best peaks to get the other interesting info
    mirnov_data = return_values(data_fft.signal,good_indices)
    power_values = np.sum(np.abs(mirnov_data),axis=1)
    #print 'power shape :', power_values.shape
    mirnov_tmp_data = np.angle(mirnov_data)
    if get_ne_array:
        ne_mode = return_values(ne_fft.signal,good_indices)
        ne_static = np.abs(return_values(ne_fft.signal,good_indices, force_index = 0))
    else:
        ne_mode = np.array([None for i in mirnov_data])
        ne_static = np.array([None for i in mirnov_data])
    if get_naked_coil:
        naked_coil_values = return_values(naked_coil_fft.signal,good_indices, force_index = 0)
    else:
        naked_coil_values =  np.array([None for i in mirnov_data])
    time_values = return_time_values(data_fft.timebase, good_indices)
    freq_values = return_non_freq_dependent(data_fft.frequency_base,good_indices)
    
    #Get other misc data
    #kh_value, ne_value, ne_time, main_current, heating_freq, ne_value_list, ne_time_list = HMA_funcs.extract_ne_kh(current_shot, ne_array = 0)
    print('getting misc data from meta data....')
    kh_value = data.meta['kh'], 
    if data.meta['heating_freq']!=None:
        heating_freq = data.meta['heating_freq']
    else:
        heating_freq = 0
        
    #Constants... shot, heating, kh
    print('kh :', kh_value)
    shot_values = time_values * 0 + current_shot
    kh_values = time_values * 0 + kh_value
    heating_values = time_values * 0 + heating_freq

    #Change Mirnov data to phase differences in [-pi,pi)
    mirnov_angles = (-np.diff(mirnov_tmp_data))%(2.*np.pi)
    mirnov_angles[mirnov_angles>np.pi] -= (2.*np.pi)
    print('make misc dictionary')
    #Put everything together for the datamining step
    misc_data_dict = {'shot':shot_values, 'kh':kh_values,'heating':heating_values,
                      'time':time_values, 'freq':freq_values, 'ne_static':ne_static, 
                      'ne_mode':ne_mode, 'naked_coil':naked_coil_values,'serial':np.arange(0,len(shot_values)),
                      'power':power_values, 'mirnov_data':mirnov_data}

    #This is legacy stuff to make it work....
    # if get_ne_array:
    #     for i in range(ne_mode.shape[1]):
    #         misc_data_dict['ne%d_omega'%(i+1)] = ne_mode[:,i]
    #         misc_data_dict['ne%d'%(i+1)] = ne_static[:,i]

    # if get_naked_coil:
    #     for i,axis in enumerate(['x','y','z']):
    #         misc_data_dict['coil_1{axis}'.format(axis=axis)] = naked_coil_values[:,i]
    if ax!=None:
        #amp = np.abs(data_fft.signal[:,0,:])
        amp = np.log(np.average(np.abs(data_fft.signal[:,:,:])**2,axis=1))
        L,R = np.min(data_fft.timebase), np.max(data_fft.timebase)
        B,T = np.min(data_fft.frequency_base), np.max(data_fft.frequency_base)
        im=ax.imshow(amp.T,aspect='auto',origin='lower',extent=[L,R,B,T])
    print('finished....', current_shot)
    return mirnov_angles, misc_data_dict

#generate the datamining object, and perform the datamining
def perform_data_datamining(mirnov_angles, misc_data_dict, n_clusters = 16, n_iterations = 60):
    feat_obj = clust.feature_object(instance_array = mirnov_angles, misc_data_dict = misc_data_dict)
    z = feat_obj.cluster(method="EM_VMM",n_clusters=n_clusters,n_iterations = n_iterations,start='k_means',verbose=0)
    #z.plot_VM_distributions()
    #z = feat_obj.cluster(method="k_means",n_clusters=n_clusters,n_iterations = n_iterations)
    #z.fit_vonMises()
    return z

def filter_by_kappa_cutoff(z, ave_kappa_cutoff=25, ax = None, prob_cutoff = None, cutoff_by='sigma_eq'):
    total_passes = 0; start = 1
    misc_data_dict2 = {}
    for i in range(z.cluster_details['EM_VMM_kappas'].shape[0]):
        ave_kappa = np.sum(z.cluster_details['EM_VMM_kappas'][i,:])/z.cluster_details['EM_VMM_kappas'].shape[1]
        std_eq, std_bar = clust.sigma_eq_sigma_bar(z.cluster_details['EM_VMM_kappas'][i,:],deg=True)


        #print i, np.sum(z.cluster_details['EM_VMM_kappas'][i,:])/z.cluster_details['EM_VMM_kappas'].shape[1],np.sum(z.cluster_assignments==i)
        current = z.cluster_assignments==i
        if ax!=None: ax.plot(z.feature_obj.misc_data_dict['time'][current],z.feature_obj.misc_data_dict['freq'][current],'k,') 
        include = 0
        if cutoff_by=='sigma_eq':
            if std_eq < ave_kappa_cutoff: include = 1
        elif cutoff_by=='sigma_bar':
            if std_bar < ave_kappa_cutoff: include = 1
        elif cutoff_by=='kappa_bar':
            if ave_kappa > ave_kappa_cutoff: include = 1

        if include:
            total_passes += np.sum(current)
            if ax!=None: ax.plot(z.feature_obj.misc_data_dict['time'][current],z.feature_obj.misc_data_dict['freq'][current],'ko') 
            #print total_passes
            if prob_cutoff!=None:
                prob_cutoff = z.cluster_details['zij'][:,i]>prob_cutoff
                current_new = current*prob_cutoff
            else:
                current_new = current
                
            #print '###############', np.sum(current), np.sum(prob_cutoff), prob_cutoff.shape, current.shape#, np.sum(current_new)
            if start==1:
                #print '################', np.sum(z.cluster_details['zij'][current,:]>0.90), np.sum(z.cluster_details['zij'][current,i]<0.90)
                instance_array2 = z.feature_obj.instance_array[current_new,:]
                start = 0
                for i in z.feature_obj.misc_data_dict.keys(): misc_data_dict2[i] = copy.deepcopy(z.feature_obj.misc_data_dict[i][current_new])
            else:
                #print '################', np.sum(z.cluster_details['zij'][current,:]>0.90), np.sum(z.cluster_details['zij'][current,i]<0.90)
                instance_array2 = np.append(instance_array2, z.feature_obj.instance_array[current_new,:],axis=0)
                for i in z.feature_obj.misc_data_dict.keys():
                    misc_data_dict2[i] = np.append(misc_data_dict2[i], z.feature_obj.misc_data_dict[i][current_new],axis=0)
    #Catch incase no good clusters were found....
    if start == 1: instance_array2 = None; misc_data_dict2 = None
    return instance_array2, misc_data_dict2

def single_shot(current_shot, array_names, NFFT, hop, n_pts, lower_freq, ax, start_time, end_time, perform_datamining, ave_kappa_cutoff, cutoff_by):
    mirnov_angles, misc_data_dict_cur = extract_data_by_picking_peaks(current_shot, array_names, NFFT=NFFT, hop=hop,n_pts=n_pts,lower_freq=lower_freq, ax = ax, time_window = [start_time, end_time])
    if perform_datamining:
        z = perform_data_datamining(mirnov_angles, misc_data_dict_cur, n_clusters = 16, n_iterations = 50)
        instance_array_cur, misc_data_dict_cur = filter_by_kappa_cutoff(z, ave_kappa_cutoff=ave_kappa_cutoff, ax = ax, cutoff_by = cutoff_by)
        return instance_array_cur, misc_data_dict_cur, z.cluster_details['EM_VMM_kappas']
    else:
        instance_array_cur = mirnov_angles
        return instance_array_cur, misc_data_dict_cur, np.zeros(5)


def single_shot_wrapper(input_data):
    #instance_array_cur, misc_data_dict_cur = single_shot(input_data[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5], input_data[6], input_data[7],input_data[8]) 
    instance_array_cur, misc_data_dict_cur, kappa_array = single_shot(*input_data)
    #print 'some kind of error occured'
    #instance_array_cur=None
    #misc_data_dict_cur=None
    return instance_array_cur, misc_data_dict_cur, kappa_array

def multi_stft(shot_selection, array_names, n_cpus=1, NFFT=2048, perform_datamining = 1, overlap=4, n_pts=20, lower_freq = 1500, filter_cutoff = 20, cutoff_by = 'sigma_eq'):
    #Get the scan details
    shot_list, start_times, end_times = H1_scan_list.return_scan_details(shot_selection)
    rep = itertools.repeat
    hop = NFFT/overlap
    input_data = itertools.izip(shot_list, rep(array_names), rep(NFFT), 
                                rep(hop), rep(n_pts), rep(lower_freq), rep(None),
                                start_times, end_times, rep(perform_datamining), 
                                rep(filter_cutoff), rep(cutoff_by))
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print('creating pool map')
        results = pool.map(single_shot_wrapper, input_data)
        print('waiting for pool to close ')
        pool.close();pool.join()
        print('pool finished')
    else:
        results = map(single_shot_wrapper, input_data)
    start=1
    #Put everything back together
    kappa_list = []
    for i,tmp in enumerate(results):
        print(i)
        kappa_list.append(copy.deepcopy(tmp[2]))
        if tmp[0]!=None:
            if start==1:
                instance_array = copy.deepcopy(tmp[0])
                misc_data_dict = copy.deepcopy(tmp[1])
                start = 0
            else:
                instance_array = np.append(instance_array, tmp[0],axis=0)
                for i in misc_data_dict.keys():
                    misc_data_dict[i] = np.append(misc_data_dict[i], tmp[1][i],axis=0)
        else:
            print('something has failed....')
    return clust.feature_object(instance_array = instance_array, misc_data_dict = misc_data_dict), kappa_list


def combine_feature_sets(feat_obj1, feat_obj2):
    feat_obj2 = copy.deepcopy(feat_obj2)
    print('reference size {}, other size {}'.format(feat_obj1.instance_array.shape,feat_obj2.instance_array.shape))

    #go through each shot....
    svd_shots = feat_obj1.misc_data_dict['shot']
    ticked_off = svd_shots * 0
    unique_shots = np.unique(svd_shots)
    total = 0
    svd_replace_keys = feat_obj1.misc_data_dict.keys()
    stft_keys = feat_obj2.misc_data_dict.keys()
    common_keys = list(set(svd_replace_keys).intersection(stft_keys))
    sample_rate = 1000000.;
    period = 0.5*1./sample_rate
    
    for shot in unique_shots:
        filt_stft = np.nonzero(shot == feat_obj2.misc_data_dict['shot'])[0]
        filt_svd = np.nonzero(shot == feat_obj1.misc_data_dict['shot'])[0]

        for i in filt_svd:
            time_difference = (np.abs(feat_obj1.misc_data_dict['time'][i] - feat_obj2.misc_data_dict['time'][filt_stft]))<(period)
            freq_difference = (np.abs(feat_obj1.misc_data_dict['freq'][i] - feat_obj2.misc_data_dict['freq'][filt_stft]))<(0.1)
            success = np.nonzero(time_difference*freq_difference)[0]
            if len(success)>1: 
                print('there may be a problem....', i, success, [filt_stft[j] for j in success])
                for j in success:
                    print(feat_obj1.misc_data_dict['time'][i], feat_obj2.misc_data_dict['time'][filt_stft[j]])
                    print(feat_obj1.misc_data_dict['freq'][i], feat_obj2.misc_data_dict['freq'][filt_stft[j]])
                    print(feat_obj1.misc_data_dict['shot'][i], feat_obj2.misc_data_dict['shot'][filt_stft[j]])

            elif len(success)==1:
                ticked_off[i]=1
                rep_ind = success[0]
                feat_obj2.instance_array[rep_ind,:] = feat_obj1.instance_array[i,:]
                for j in common_keys:
                    feat_obj2.misc_data_dict[j][rep_ind] = feat_obj1.misc_data_dict[j][i]
                total+=len(success)

    print('total replaced : {}'.format(total))

    extras = np.nonzero(ticked_off==0)[0]
    print('total not replaced : {}'.format(len(extras)))
    for i in extras:
        feat_obj2.instance_array = np.append(feat_obj2.instance_array, (feat_obj1.instance_array[i,:])[np.newaxis,:],axis=0)
        for j in common_keys:
           feat_obj2.misc_data_dict[j] =  np.append(feat_obj2.misc_data_dict[j], feat_obj1.misc_data_dict[j][i])



    print('Output size : {}'.format(feat_obj2.instance_array.shape))
    return feat_obj2
