'''
SH : 2May2013
bdb lots of mods 
testing: pyfusion/examples/clusterDA.py
See also clustering/EM_vonMises.py for another attempt
Lots of duplication - needs rationalisation
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as pt
import math, time, copy, itertools, multiprocessing, os
from sklearn import mixture
from scipy.cluster import vq
from scipy.stats.distributions import vonmises
from scipy.stats.distributions import norm
import pyfusion
from pyfusion.debug_ import debug_

import sys
if sys.version < "3":
    import cPickle as pickle
    izip = itertools.izip
else:
    import pickle
    izip = zip

import scipy.special as spec
import scipy.optimize as opt

pilims = [-np.pi,np.pi]
pilims = [-3.3,3.3]  # make more room for the lables 3 and -3
colours0 = ['r','k','b','y','m']   # SHauns original list for phase_vs_phase
colours1 = ['r','k','b','y','m','g','c','orange','purple','lightgreen','lightgray'] # extended


def compare_several_kappa_values(clusters, pub_fig = 0, alpha = 0.05,decimation=10, labels=None, plot_style_list=None, filename='extraction_comparison.pdf',xaxis='sigma_eq',max_cutoff_value = 35, vline = None):
    '''xaxis can be sigma_eq, kappa_bar or sigma_bar
    '''
    fig, ax = pt.subplots()
    if pub_fig:
        cm_to_inch=0.393701
        import matplotlib as mpl
        old_rc_Params = mpl.rcParams
        mpl.rcParams['font.size']=8.0
        mpl.rcParams['axes.titlesize']=8.0#'medium'
        mpl.rcParams['xtick.labelsize']=8.0
        mpl.rcParams['ytick.labelsize']=8.0
        mpl.rcParams['lines.markersize']=5.0
        mpl.rcParams['savefig.dpi']=150
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*0.8*cm_to_inch)

    for cur_cluster,cur_label,plot_style in zip(clusters, labels,plot_style_list):
        std_bar, std_eq = sigma_eq_sigma_bar(cur_cluster.cluster_details["EM_VMM_kappas"], deg=True)
        averages = np.average(cur_cluster.cluster_details["EM_VMM_kappas"],axis=1)
        items = []
        kappa_cutoff_list = range(max_cutoff_value)
        for kappa_cutoff_tmp in kappa_cutoff_list:
            if xaxis=='sigma_eq':
                cluster_list = np.arange(len(averages))[std_eq<kappa_cutoff_tmp]
            elif xaxis=='sigma_bar':
                cluster_list = np.arange(len(averages))[std_bar<kappa_cutoff_tmp]
            else:
                cluster_list = np.arange(len(averages))[averages>kappa_cutoff_tmp]
            total = 0
            for i in cluster_list: total+= np.sum(cur_cluster.cluster_assignments==i)
            items.append(total)
        ax.plot(kappa_cutoff_list, items,plot_style,label=cur_label)
    ax.legend(loc='best',prop={'size':8.0})
    ax.set_ylabel('Number of Features')
    if vline!=None:
        ax.vlines(vline,ax.get_ylim()[0], ax.get_ylim()[1])
    if xaxis=='sigma_eq':
        ax.set_xlabel(r'$\sigma_{eq}$ cutoff')
    elif xaxis=='sigma_bar':
        ax.set_xlabel(r'$\bar{\sigma}$ cutoff')
    else:
        ax.set_xlabel(r'$\kappa$ cutoff')
    #fig.savefig('hello.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    fig.canvas.draw(); fig.show()

def sigma_eq_sigma_bar(kappas, deg=False):
    std_circ = convert_kappa_std(kappas, deg=False)
    if len(kappas.shape)==2:
        std_bar = np.mean(std_circ,axis=1)
        std_eq = (np.product(std_circ,axis=1))**(1./kappas.shape[1])
    elif len(kappas.shape)==1:
        std_bar = np.mean(std_circ)
        std_eq = (np.product(std_circ))**(1./len(kappas))
    else:
        raise ValueError("kappa is a strange dimension")
    #print std_eq.shape, kappas.shape
    #print std_eq, std_bar
    if deg:
        return std_bar*180./np.pi, std_eq*180./np.pi
    else:
        return std_bar, std_eq

def compare_several_clusters(clusters, pub_fig = 0, alpha = 0.05,decimation=10, labels=None, filename='hello.pdf', kappa_ref_cutoff=0, plot_indices = [0,1], colours = None, markers = None):
    '''
    Clusters contains a list of clusters
    Print out a comparison between two sets of clusters
    to compare datamining methods...

    SH : 7May2013
    '''
    if pub_fig:
        cm_to_inch=0.393701
        import matplotlib as mpl
        old_rc_Params = mpl.rcParams
        mpl.rcParams['font.size']=8.0
        mpl.rcParams['axes.titlesize']=8.0#'medium'
        mpl.rcParams['xtick.labelsize']=8.0
        mpl.rcParams['ytick.labelsize']=8.0
        mpl.rcParams['lines.markersize']=1.0
        mpl.rcParams['savefig.dpi']=100

    reference_cluster = clusters[0]
    clusters1 = list(set(reference_cluster.cluster_assignments))
    averages = np.average(reference_cluster.cluster_details["EM_VMM_kappas"],axis=1)
    print(clusters1)
    print(averages)
    clusters1 = np.array(clusters1)[averages > kappa_ref_cutoff]
    print(clusters1)
    ordering_list = [clusters1]
    string_list = []
    for test_cluster,cur_label in zip(clusters[1:],labels[1:]):
        print(test_cluster.settings['method'])
        #clusters1 = list(set(reference_cluster.cluster_assignments))
        clusters2 = list(set(test_cluster.cluster_assignments))
        similarity = np.zeros((len(clusters1),len(clusters2)),dtype=int)
        for i,c1 in enumerate(clusters1):
            for j,c2 in enumerate(clusters2):
                #tmp1 = (cluster1.cluster_assignments==c1)
                #similarity[i,j]=np.sum(cluster2.cluster_assignments[tmp1]==c2)
                similarity[i,j]=np.sum((reference_cluster.cluster_assignments==c1) * (test_cluster.cluster_assignments==c2))
                # np.sum((cluster1.cluster_assignments==c1) == (cluster2.cluster_assignments==c2))
        print('    %7s'%('clust') + ''.join(['%7d'%j2 for j1,j2 in enumerate(clusters2)]))
        for i,clust_num in enumerate(clusters1):
            print('ref %7d'%(clust_num,) + ''.join(['%7d'%j for j in similarity[i,:]]))
            #print '%5d'.join(map(int, similarity[i,:]))
        tmp1 = float(np.sum(np.max(similarity,axis=1)))
        data_points = len(reference_cluster.cluster_assignments)
        print('correct:{}, false:{}'.format(int(tmp1), data_points-int(tmp1)))
        string_list.append([cur_label,'correct_percentage:{:.2f}'.format(tmp1/np.sum(similarity)*100)])
        print('correct_percentage:{:.2f}'.format(tmp1/np.sum(similarity)*100))
        print('false_percentage:{:.2f}'.format(100-tmp1/np.sum(similarity)*100))
        #best_match_for_c1 = np.argmax(similarity,axis=1)
        #best_match_for_c2 = np.argmax(similarity,axis=0)

        #truth = (similarity==similarity)
        order = np.zeros(similarity.shape[0],dtype=int)
        for i in range(similarity.shape[0]):
            index = np.argmax(similarity)
            row_index, col_index = np.unravel_index(similarity.argmax(), similarity.shape)
            print(col_index, row_index)
            #col_index = index %similarity.shape[0]
            #row_index = index/similarity.shape[0]
            order[row_index] = col_index
            similarity[row_index,:]=0
            similarity[:,col_index]=0
            print(order)
        ordering_list.append(order)
    n_plots = len(clusters)
    ncols = 2
    nrows = n_plots/2
    if (n_plots - (nrows * ncols))>0.01: nrows+=1
    print(n_plots, ncols, nrows)
    print(clusters)
    fig, ax = pt.subplots(ncols=ncols,nrows=nrows, sharex = 'all', sharey = 'all')
    ax = ax.flatten()
    if pub_fig:
        fig.set_figwidth(8.48*cm_to_inch)
        fig.set_figheight(8.48*(0.5*nrows)*cm_to_inch)

    instance_array = reference_cluster.feature_obj.instance_array % (2.*np.pi)
    instance_array[instance_array>np.pi]-=(2.*np.pi)
    print('####################')
    for i in string_list: print(i)
    print('####################')
    for test_cluster, cur_ax, order, index in zip(clusters,ax, ordering_list, range(len(ordering_list))):
        print('hello', order)
        cluster_list = list(set(test_cluster.cluster_assignments))
        n_dimensions = instance_array.shape[1]
        if (colours is None) or (markers is None):
            colours_base = ['r','k','b','y','m']
            marker_base = ['o','x','+','s','*']
            markers = []; colours = []
            for i in marker_base:
                markers.extend([i for j in colours_base])
                colours.extend(colours_base)
        for ref,cur in enumerate(order):
            #print i, colours[ref],markers[ref]
            cluster = cluster_list[order[ref]]
            current_items = test_cluster.cluster_assignments==cluster
            datapoints = instance_array[current_items,:]
            cur_ax.scatter(datapoints[::decimation,plot_indices[0]], datapoints[::decimation,plot_indices[1]],c=colours[ref],marker=markers[ref], alpha=alpha,rasterized=True,edgecolors=colours[ref])
            if labels is None:
                cur_label = test_cluster.settings['method']
            else:
                cur_label = labels[index]
        cur_ax.text(0,0,cur_label, horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.5))
            #cur_ax.plot(cluster_means[i,0],cluster_means[i,1],colours[i]+markers[i],markersize=8)
    ax[-1].set_xlim(pilims)
    ax[-1].set_ylim(pilims)

    fig.text(0.5, 0.01, r'$\Delta \psi_{{{}}}$'.format(plot_indices[0]), ha='center', va='center', fontsize = 10)
    fig.text(0.01, 0.5, r'$\Delta \psi_{{{}}}$'.format(plot_indices[1]), ha='center', va='center', rotation='vertical', fontsize=10)
    
    #fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.02, wspace=0.01)#,left=0.10, bottom=0.10,top=0.95, right=0.95)
    #fig.savefig('hello.eps', bbox_inches='tight', pad_inches=0)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)

    fig.canvas.draw(); fig.show()
    return fig, ax


def compare_two_cluster_results(cluster1, cluster2):
    '''
    Print out a comparison between two sets of clusters
    to compare datamining methods...

    SH : 7May2013
    '''
    clusters1 = list(set(cluster1.cluster_assignments))
    clusters2 = list(set(cluster2.cluster_assignments))
    similarity = np.zeros((len(clusters1),len(clusters2)),dtype=int)
    for i,c1 in enumerate(clusters1):
        for j,c2 in enumerate(clusters2):
            #tmp1 = (cluster1.cluster_assignments==c1)
            #similarity[i,j]=np.sum(cluster2.cluster_assignments[tmp1]==c2)
            similarity[i,j]=np.sum((cluster1.cluster_assignments==c1) * (cluster2.cluster_assignments==c2))
            # np.sum((cluster1.cluster_assignments==c1) == (cluster2.cluster_assignments==c2))
    print('%7s'%('clust') + ''.join(['%7d'%j2 for j1,j2 in enumerate(clusters2)]))
    for i,clust_num in enumerate(clusters1):
        print('%7d'%(clust_num,) + ''.join(['%7d'%j for j in similarity[i,:]]))
        #print '%5d'.join(map(int, similarity[i,:]))
    best_match_for_c1 = np.argmax(similarity,axis=1)
    best_match_for_c2 = np.argmax(similarity,axis=0)

    #print np.argmax(similarity,axis=1)

    n_clusters = len(clusters1)
    n_cols = int(math.ceil(n_clusters**0.5))
    kh_plot_item = 'kh'
    freq_plot_item = 'freq'
    if n_clusters/float(n_cols)>n_clusters/n_cols:
        n_rows = n_clusters/n_cols + 1
    else:
        n_rows = n_clusters/n_cols
    #n_rows = 4; n_cols = 4
    fig, ax = pt.subplots(nrows = n_rows, ncols = n_cols, sharex = 'all', sharey='all'); ax = ax.flatten()
    fig2, ax2 = pt.subplots(nrows = n_rows, ncols = n_cols, sharex = 'all', sharey='all'); ax2 = ax2.flatten()
    for i,cluster,best_match in zip(range(len(clusters1)),clusters1,best_match_for_c1):
        current_items1 = cluster1.cluster_assignments==cluster
        current_items2 = cluster2.cluster_assignments==best_match
        both = current_items2*current_items1
        if np.sum(current_items1)>10:
            ax[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][current_items1]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][current_items1])/1000., s=80, c='b',  marker='o', norm=None, alpha=0.02)
            ax[i].scatter((cluster2.feature_obj.misc_data_dict[kh_plot_item][current_items2]), (cluster2.feature_obj.misc_data_dict[freq_plot_item][current_items2])/1000., s=80, c='r',  marker='o', norm=None, alpha=0.02)
            ax[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][both]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][both])/1000., s=80, c='k',  marker='o', norm=None, alpha=0.02)
    for i,cluster,best_match in zip(range(len(clusters2)),clusters2,best_match_for_c2):
        current_items2 = cluster2.cluster_assignments==cluster
        current_items1 = cluster1.cluster_assignments==best_match
        both = current_items2*current_items1
        if np.sum(current_items1)>10:
            ax2[i].scatter((cluster2.feature_obj.misc_data_dict[kh_plot_item][current_items2]), (cluster2.feature_obj.misc_data_dict[freq_plot_item][current_items2])/1000., s=80, c='b',  marker='o', norm=None, alpha=0.02)
            ax2[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][current_items1]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][current_items1])/1000., s=80, c='r',  marker='o', norm=None, alpha=0.02)
            ax2[i].scatter((cluster1.feature_obj.misc_data_dict[kh_plot_item][both]), (cluster1.feature_obj.misc_data_dict[freq_plot_item][both])/1000., s=80, c='k',  marker='o', norm=None, alpha=0.02)

    ax[-1].set_xlim([0.201,0.99])
    ax[-1].set_ylim([0.1,99.9])
    fig.suptitle('blue cluster1, red best match from cluster2, black common to both')
    fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
    fig.canvas.draw(); fig.show()
    ax2[-1].set_xlim([0.201,0.99])
    ax2[-1].set_ylim([0.1,99.9])
    fig2.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
    fig2.suptitle('blue cluster2, red best match from cluster1, black common to both')
    fig2.canvas.draw(); fig2.show()
    return similarity


# this 'constant' is defined for convenience when using convert_DA_file
default_correspondence = 'indx,serial t_mid,time amp,RMS freq,freq p,p a12,a12, shot,shot k_h,kh, ne_1,ne1, ne_2,ne2 ne_3,ne3 ne_4,ne4 ne_5,ne5 ne_6,ne6 ne_7,ne7 b_0,b_0 p_rf,p_rf'
def convert_DA_file(filename, correspondence=default_correspondence, debug=1, limit=None, Kilohertz=1, load_all=False, keysel=None):
    """ Converts a DA_datamining file to a form compatible with this package.
    returns(instance_array, misc_data) with names converted according to 
    correspondence, input as pairs separated by spaces.
    
    limit = 10000  selects ~10000 samples randowmly ( -10000 for repeatable sec)
    load_all = True   will load all misc data, even those not in the 
    correspondence list.
    Reverse conversion (Shauns to DA) features are built into DA-datamining
    e.g. 
    fo = co23.feature_obj
    dd = dict(phases=fo.instance_array)
    dd.update(fo.misc_data_dict) # should check that the dimensions agree
    DA23 = DA(dd) 
    
    """
    from pyfusion.data.DA_datamining import DA
    pairs = correspondence.split(' ')
    corr_dict = {}
    for pair in pairs:
        corr_dict.update({pair.split(',')[0]: pair.split(',')[1]})

    # don't save DA if we are taking all - wasteful of space
    if keysel is None: ddin = DA(filename, load=1, limit=limit).da
    else:  # this should be rationalised
        print('selecting {n} instances'. format(n=len(keysel)))
        DAsel = DA(filename)
        DAsel.load(sel=keysel)
        ddin = DAsel.copyda()

    inst_arr = ddin.pop('phases')
    if load_all:
        dd = ddin
    else:
        dd = {}
    print('corr_dict={}'.format(corr_dict))    
    for k in list(ddin.keys()):
        if k in list(corr_dict):
            dd.update({corr_dict[k]:ddin.pop(k)})

    if 'freq' in dd: dd['freq'] = 1000*np.array(dd['freq'])
    misc_data = dd
    return(inst_arr, misc_data)

class feature_object():
    '''
    This is suposed to be the feature object
    SH : 6May2013
    '''
    def __init__(self, instance_array=None, misc_data_dict=None, filename = None):#, misc_data_labels):
        '''
        feature_object... this contains all of the raw data that is a 
        result of feature extraction. It can be initialised by passing
        an instance_array and misc_data_dict dictionary, or alternatively,
        the filename of a pickle file that was saved with 
        feature_object.dump_data()
        '''
        if instance_array is None and filename!=None:
            self.load_data(filename)
        else:
            self.instance_array = instance_array
            self.misc_data_dict = misc_data_dict
            self.clustered_objects = []

    def cluster(self,**kwargs):
        '''This method will perform clustering using one of the
        following methods: k-means (scikit-learn), Expectation
        maximising using a Gaussian mixture model EM_GMM
        (scikit-learn) k_means_periodic (SH implementation) Expecation
        maximising using a von Mises mixture model EM_VMM (SH
        implementation)

        **kwargs: method : To determine which clustering algorithm to
        use can be : k-means, EMM_GMM, k_means_periodic, EM_VMM

        Other kwargs to overide the following default settings for
        each clustering algorithmn

        'k_means': {'n_clusters':9, 'sin_cos':1,
        'number_of_starts':30}, 'EM_GMM' : {'n_clusters':9,
        'sin_cos':1, 'number_of_starts':30}, 'k_means_periodic' :
        {'n_clusters':9, 'number_of_starts':10, 'n_cpus':1,
        'distance_calc':'euclidean','convergence_diff_cutoff': 0.2,
        'iterations': 40, 'decimal_roundoff':2}, 'EM_VMM' :
        {'n_clusters':9, 'n_iterations':20, 'n_cpus':1}}

        returns a cluster object that also gets appended to the
        self.clustered_objects list

        SH: 6May2013
        '''
        self.clustered_objects.append(clusterer_wrapper(self,**kwargs))
        return self.clustered_objects[-1]


    def dump_data(self,filename):
        '''
        This is saves the important parts of the clustering data
        It does not save the object itself!!!

        The idea here is that we can save the data, and when we reload it,
        we have access to any new features.

        SH: 8May2013
        '''
        dump_dict = {}
        dump_dict['instance_array'] = self.instance_array
        dump_dict['misc_data_dict'] = self.misc_data_dict
        dump_dict['clustered_objects']={}
        clust_objs = dump_dict['clustered_objects']
        for i,tmp_clust in enumerate(self.clustered_objects):
            clust_objs[i]={}
            clust_objs[i]['settings']=tmp_clust.settings
            clust_objs[i]['cluster_assignments']=tmp_clust.cluster_assignments
            clust_objs[i]['cluster_details']=tmp_clust.cluster_details
        pickle.dump(dump_dict,file(filename,'w'))

    def load_data(self,filename):
        '''
        This is for loading saved clustering data
        SH: 8May2013
        '''
        dump_dict = pickle.load(file(filename,'r'))
        self.instance_array = dump_dict['instance_array']
        self.misc_data_dict = dump_dict['misc_data_dict']
        self.clustered_objects = []
        clust_objs = dump_dict['clustered_objects']
        for i in clust_objs.keys():
            tmp = clustering_object()
            tmp.settings = clust_objs[i]['settings']
            tmp.cluster_assignments = clust_objs[i]['cluster_assignments']
            tmp.cluster_details = clust_objs[i]['cluster_details']
            tmp.feature_obj = self
            self.clustered_objects.append(tmp)

    def print_cluster_details(self,):
        for i,clust in enumerate(self.clustered_objects):
            print(i, clust.settings)
            

        
class clustering_object():
    '''Generic clustering_object, this will have the following
    attributes instance_array : array of phase differences

    SH : 6May2013 '''

    def members(self, cl_num=None):
        """ return the indices in the instance array corresponding to cluster i
        """
        num = len(self.cluster_details['EM_VMM_kappas'])
        if not cl_num in range(num):
            raise LookupError('cluster number {c} not between 0 and {n} inclusive'
                              .format(c=cl_num, n=num-1))
        return(np.where(self.cluster_assignments == cl_num)[0])

    def make_mode_list(self, min_kappa = 4, plot=False):
        """ Return a mode_list (such as used in new_mode_identify_script from 
        a cluster_object,
        ps: this is another reason for make new_modeidentifier _script a class.
        """
        from pyfusion.clustering.modes import Mode
        mode_list = []
        means = self.cluster_details['EM_VMM_means']
        kappas = self.cluster_details['EM_VMM_kappas']

        for (i,k) in enumerate(kappas):
            if np.min(k)>min_kappa:
                mode_list.append(Mode('cl{n}'.format(n=i), -99, -00, 
                                      means[i], np.sqrt(1/k),id=i))

        if plot: 
            for (i,mode) in enumerate(mode_list):
                ncols = 1+np.sqrt(len(mode_list))
                pt.subplot(ncols, ncols, i+1)
                mode.plot()
        return(mode_list)

    def plot_kh_freq_all_clusters(self,color_by_cumul_phase = 1):
        '''plot kh vs frequency for each cluster - i.e looking for
        whale tails The colouring of the points is based on the total
        phase along the array i.e a 1D indication of the clusters

        SH: 9May2013

        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        kh_plot_item = 'kh'
        freq_plot_item = 'freq'
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = len(cluster_list)
        fig_kh, ax_kh = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        misc_data_dict = self.feature_obj.misc_data_dict
        if color_by_cumul_phase:
            instance_array2 = modtwopi(self.feature_obj.instance_array, offset=0)
            max_lim = -1*2.*np.pi; min_lim = (-3.*2.*np.pi)
            total_phase = np.sum(instance_array2,axis=1)
            print(np.max(total_phase), np.min(total_phase))
            total_phase = np.clip(total_phase,min_lim, max_lim)
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                if color_by_cumul_phase:
                    ax_kh[cluster].scatter((misc_data_dict[kh_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items])/1000., s=80, c=total_phase[current_items], vmin = min_lim, vmax = max_lim, marker='o', cmap='jet', norm=None, alpha=0.2)
                else:
                    ax_kh[cluster].scatter((misc_data_dict[kh_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items])/1000, s=100, c='b', marker='o', cmap=None, norm=None)
                ax_kh[cluster].legend(loc='best')
        ax_kh[-1].set_xlim([0.201,0.99])
        ax_kh[-1].set_ylim([0.1,199.9])
        fig_kh.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig_kh.suptitle(suptitle,fontsize=8)
        fig_kh.canvas.draw(); fig_kh.show()
        return fig_kh, ax_kh


    def plot_time_freq_all_clusters(self,color_by_cumul_phase = 1):
        '''plot kh vs frequency for each cluster - i.e looking for whale tails
        The colouring of the points is based on the total phase along the array
        i.e a 1D indication of the clusters

        SH: 9May2013

        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        time_plot_item = 'time'
        freq_plot_item = 'freq'
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = len(cluster_list)
        fig_kh, ax_kh = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        misc_data_dict = self.feature_obj.misc_data_dict
        if color_by_cumul_phase:
            instance_array2 = modtwopi(self.feature_obj.instance_array, offset=0)
            max_lim = -1*2.*np.pi; min_lim = (-3.*2.*np.pi)
            total_phase = np.sum(instance_array2,axis=1)
            print(np.max(total_phase), np.min(total_phase))
            total_phase = np.clip(total_phase,min_lim, max_lim)
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                if color_by_cumul_phase:
                    ax_kh[cluster].scatter((misc_data_dict[time_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items])/1000., s=20, c=total_phase[current_items], vmin = min_lim, vmax = max_lim, marker='o', cmap='jet', norm=None, alpha=0.8)
                else:
                    ax_kh[cluster].scatter((misc_data_dict[time_plot_item][current_items]), (misc_data_dict[freq_plot_item][current_items])/1000, s=100, c='b', marker='o', cmap=None, norm=None)
                ax_kh[cluster].legend(loc='best')
        ax_kh[-1].set_xlim([0.,0.1])
        ax_kh[-1].set_ylim([0.,150.])
        fig_kh.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig_kh.suptitle(suptitle,fontsize=8)
        fig_kh.canvas.draw(); fig_kh.show()
        return fig_kh, ax_kh

    def fit_vonMises(self,):
        instance_array = self.feature_obj.instance_array
        mu_list = np.ones((len(set(self.cluster_assignments)),self.feature_obj.instance_array.shape[1]),dtype=float)
        kappa_list = mu_list*1.
        self.cluster_details['EM_VMM_means'], self.cluster_details['EM_VMM_kappas'],tmp1,tmp2 = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, self.feature_obj.instance_array, self.cluster_assignments)

    def plot_VM_distributions(self,):
        '''Plot the vonMises distributions for each dimension for each cluster
        Also plot the histograms - these are overlayed with dashed lines

        SH: 9May2013
        '''
        #Plot the two distributions over each other to check for goodness of fit
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        cluster_list = list(set(self.cluster_assignments))
        n_clusters = np.max([len(cluster_list),np.max(cluster_list)])
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        delta = 300
        x = np.linspace(-np.pi, np.pi, delta)
        instance_array = self.feature_obj.instance_array
        try:
            cluster_mu = self.cluster_details['EM_VMM_means']
            cluster_kappa = self.cluster_details['EM_VMM_kappas']
        except KeyError:
            print('EM_VMM cluster details not available - calculating them')
            self.fit_vonMises()
        cluster_mu = self.cluster_details['EM_VMM_means']
        cluster_kappa = self.cluster_details['EM_VMM_kappas']
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                for dimension in range(instance_array.shape[1]):
                    #print cluster, dimension
                    kappa_tmp = cluster_kappa[cluster][dimension]
                    loc_tmp = cluster_mu[cluster][dimension]
                    fit_dist_EM = vonmises(kappa_tmp,loc_tmp)
                    Z_EM = fit_dist_EM.pdf(x)
                    if np.sum(np.isnan(Z_EM))==0:
                        tmp = ax[cluster].plot(x,Z_EM,'-')
                        current_color = tmp[0].get_color()
                        ax[cluster].text(x[np.argmax(Z_EM)],np.max(Z_EM),str(dimension))
                    bins = np.linspace(-np.pi,np.pi,360)
                    histogram_data = (instance_array[current_items,dimension]) %(2.*np.pi)
                    histogram_data[histogram_data>np.pi]-=(2.*np.pi)
                    tmp3 = np.histogram(histogram_data,bins = bins,range=pilims)
                    dx = tmp3[1][1]-tmp3[1][0]
                    integral = np.sum(dx*tmp3[0])
                    if np.sum(np.isnan(tmp3[0]/integral))==0:
                        ax[cluster].plot(tmp3[1][:-1]+dx/2, tmp3[0]/integral, marker=',',linestyle="--",color=current_color)
        ax[-1].set_xlim(pilims)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw();fig.show()
        return fig, ax

    def plot_dimension_histograms(self,pub_fig = 0, filename='plot_dim_hist.pdf',specific_dimensions = None, extra_txt_labels = '', label_loc = [-2,1.5], ylim = None):
        '''For each dimension in the data set, plot the histogram of the phase differences
        Overlay the vonMises mixture model along with the individual vonMises distributions 
        from each cluster

        SH: 9May2013
        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        instance_array = self.feature_obj.instance_array
        dimensions = instance_array.shape[1]
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')

        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300
        if specific_dimensions is None:
            specific_dimensions = range(instance_array.shape[1])
        fig, ax = make_grid_subplots(len(specific_dimensions), sharex = 'all', sharey = 'all')
        if pub_fig:
            fig.set_figwidth(8.48*cm_to_inch)
            fig.set_figheight(8.48*0.8*cm_to_inch)
        for i,dim in enumerate(specific_dimensions):
            histogram_data = (instance_array[:,dim]) %(2.*np.pi)
            histogram_data[histogram_data>np.pi]-=(2.*np.pi)
            ax[i].hist(histogram_data,bins=180,normed=True,histtype='stepfilled',range=pilims)
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw(); fig.show()
        if self.cluster_assignments!=None:
            cluster_list = list(set(self.cluster_assignments))
        delta = 300
        x = np.linspace(-np.pi, np.pi, delta)
        cluster_prob_list = []
        for cluster in cluster_list:
            cluster_prob_list.append(float(np.sum(self.cluster_assignments==cluster))/float(len(self.cluster_assignments)))
        #print cluster_prob_list
        try:
            cluster_mu = self.cluster_details['EM_VMM_means']
            cluster_kappa = self.cluster_details['EM_VMM_kappas']
        except KeyError:
            print('EM_VMM cluster details not available - calculating them')
            self.fit_vonMises()
        cluster_mu = self.cluster_details['EM_VMM_means']
        cluster_kappa = self.cluster_details['EM_VMM_kappas']
        text_labels = []
        for i,dimension in enumerate(specific_dimensions):
            cluster_sum = x*0
            for cluster, cluster_prob in zip(cluster_list, cluster_prob_list):
                current_items = (self.cluster_assignments==cluster)
                kappa_tmp = cluster_kappa[cluster][dimension]
                loc_tmp = cluster_mu[cluster][dimension]
                fit_dist_EM = vonmises(kappa_tmp,loc_tmp)
                Z_EM = cluster_prob * fit_dist_EM.pdf(x)
                cluster_sum = Z_EM + cluster_sum
                if pub_fig:
                    tmp = ax[i].plot(x,Z_EM,'-',linewidth=0.8)
                else:
                    tmp = ax[i].plot(x,Z_EM,'-',linewidth=2)
            if pub_fig:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=2)
            else:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=4)
            print('{area},'.format(area = np.sum(cluster_sum*(x[1]-x[0]))), end=' ')
            ax[i].text(label_loc[0], label_loc[1],r'$\Delta \psi_%d$ '%(dimension+1,) + extra_txt_labels, fontsize = 8)#,bbox=dict(facecolor='white', alpha=0.5))
            ax[i].locator_params(nbins=7)
        print('')
        if pub_fig:
            ax[0].set_xlim([-np.pi, np.pi])
            if ylim!=None:
                ax[0].set_ylim(ylim)
            fig.text(0.5, 0.01, r'$\Delta \psi$ (rad)', ha='center', va='center', fontsize = 10)
            fig.text(0.01, 0.5, 'Probability density', ha='center', va='center', rotation='vertical', fontsize=10)
            fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
            fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)
        ax[-1].set_xlim(pilims)
        fig.suptitle(suptitle,fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax



    def plot_dimension_histograms_amps(self,pub_fig = 0, filename='plot_dim_hist.pdf',specific_dimensions = None):
        '''For each dimension in the data set, plot the histogram of the phase differences
        Overlay the vonMises mixture model along with the individual vonMises distributions 
        from each cluster

        SH: 9May2013
        '''
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        instance_array_amps = np.abs(self.feature_obj.misc_data_dict['mirnov_data'])
        norm_factor = np.sum(instance_array_amps,axis=1)
        instance_array_amps = instance_array_amps/norm_factor[:,np.newaxis]
        dimensions = instance_array_amps.shape[1]
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')

        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300
        if specific_dimensions is None:
            specific_dimensions = range(instance_array_amps.shape[1])
        fig, ax = make_grid_subplots(len(specific_dimensions), sharex = 'all', sharey = 'all')
        if pub_fig:
            fig.set_figwidth(8.48*cm_to_inch)
            fig.set_figheight(8.48*0.8*cm_to_inch)
        for i,dim in enumerate(specific_dimensions):
            histogram_data = instance_array_amps[:,dim]
            ax[i].hist(histogram_data,bins=180,normed=True,histtype='stepfilled',range=[0,1])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw(); fig.show()
        cluster_list = list(set(self.cluster_assignments))
        delta = 300
        x = np.linspace(0, 1, delta)
        cluster_prob_list = []
        for cluster in cluster_list:
            cluster_prob_list.append(float(np.sum(self.cluster_assignments==cluster))/float(len(self.cluster_assignments)))

        cluster_means = self.cluster_details['EM_GMM_means']
        cluster_std = self.cluster_details['EM_GMM_std']
        for i,dimension in enumerate(specific_dimensions):
            cluster_sum = x*0
            for cluster, cluster_prob in zip(cluster_list, cluster_prob_list):
                current_items = (self.cluster_assignments==cluster)
                std_tmp = cluster_std[cluster][dimension]
                mean_tmp = cluster_means[cluster][dimension]
                fit_dist_GM = norm(loc=mean_tmp,scale=std_tmp)
                #fit_dist_GM = norm(std_tmp,mean_tmp)
                Z_GM = cluster_prob * fit_dist_GM.pdf(x)
                cluster_sum = Z_GM + cluster_sum
                if pub_fig:
                    tmp = ax[i].plot(x,Z_GM,'-',linewidth=0.8)
                else:
                    tmp = ax[i].plot(x,Z_GM,'-',linewidth=2)
            if pub_fig:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=2)
            else:
                tmp = ax[i].plot(x,cluster_sum,'-',linewidth=4)
            print('{area},'.format(area = np.sum(cluster_sum*(x[1]-x[0]))), end=' ')
        print('')
        if pub_fig:
            ax[0].set_xlim([-np.pi, np.pi])
            fig.text(0.5, 0.01, 'Amp', ha='center', va='center', fontsize = 10)
            fig.text(0.01, 0.5, 'Probability density', ha='center', va='center', rotation='vertical', fontsize=10)
            fig.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
            fig.tight_layo()
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)
        ax[-1].set_xlim([0,1])
        fig.suptitle(suptitle,fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax



    def plot_phase_vs_phase(self,pub_fig = 0, filename = 'phase_vs_phase.pdf',compare_dimensions=None, kappa_ave_cutoff=0, plot_means = 0, alpha = 0.05, decimation = 1, limit=None, xlabel_loc = 3, ylabel_loc = 3, colours=None):
        '''
        SH: 9May2013

        '''
        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=100

        sin_cos = 0
        if (self.settings['method'] == 'EM_VMM') or (self.settings['method'] == 'EM_VMM_soft'): 
            cluster_means = self.cluster_details['EM_VMM_means']
        elif self.settings['method'] == 'k_means':
            if self.settings['sin_cos'] == 1:
                cluster_means = self.cluster_details['k_means_centroids_sc']
                sin_cos = 1
            else:
                cluster_means = self.cluster_details['k_means_centroids']
        elif self.settings['method'] == 'k_means_periodic':
            cluster_means = self.cluster_details['k_means_periodic_centroids']
        elif self.settings['method'] == 'EM_GMM':    
            if self.settings['sin_cos'] == 1:
                cluster_means = self.cluster_details['EM_GMM_means_sc']
                sin_cos = 1
            else:
                cluster_means = self.cluster_details['EM_GMM_means']

        if limit is not None:  # limit overrides decimate
            decimation = max(self.feature_obj.instance_array.shape[0]/limit , 1)
        if sin_cos:
            instance_array = np.zeros((self.feature_obj.instance_array.shape[0],self.feature_obj.instance_array.shape[1]*2),dtype=float)
            instance_array[:,::2]=np.cos(self.feature_obj.instance_array)
            instance_array[:,1::2]=np.sin(self.feature_obj.instance_array)
            pass
        else:
            instance_array = (self.feature_obj.instance_array)%(2.*np.pi)
            instance_array[instance_array>np.pi]-=(2.*np.pi)
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        cluster_list = list(set(self.cluster_assignments))
        if colours is None:
            colours = colours1 #['r','k','b','y','m','g','c','w','orange','purple','lightgreen'] # 'r','k','b','y','m']
        marker = ['o' for i in colours]
        colours.extend(colours)
        marker.extend(['x' for i in colours])

        n_dimensions = instance_array.shape[1]
        if compare_dimensions is None:
            dims = []
            for dim in range(n_dimensions-1):
                dims.append([dim, dim+1])
        else:
            dims = compare_dimensions
        n_dimensions = instance_array.shape[1]
        fig_kh, ax_kh = make_grid_subplots(len(dims), sharex = 'all', sharey = 'all')
        if pub_fig:
            fig_kh.set_figwidth(8.48*cm_to_inch)
            fig_kh.set_figheight(8.48*0.8*cm_to_inch)
        for ax_loc,(dim1,dim2) in enumerate(dims):
            print(dim1, dim2)
            counter = 0
            for i,cluster in enumerate(cluster_list):
                if np.average(self.cluster_details["EM_VMM_kappas"][i,:])>kappa_ave_cutoff:
                    current_items = self.cluster_assignments==cluster
                    datapoints = instance_array[current_items,:]
                    ax_kh[ax_loc].scatter(datapoints[::decimation,dim1], datapoints[::decimation,dim2],c=colours[counter],marker=marker[counter], alpha=alpha,rasterized=True, edgecolors=colours[counter])
                    if plot_means: ax_kh[ax_loc].plot(cluster_means[i,dim1],cluster_means[i,dim2],colours[i]+marker[i],markersize=8)
                    counter = (counter+1) % len(colours)
            ax_kh[ax_loc].text(xlabel_loc,ylabel_loc,r'$\Delta \psi_{{{}}}$, vs $\Delta \psi_{{{}}}$'.format(dim1+1,dim2+1), horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.5))

        fig_kh.text(0.5, 0.01, r'$\Delta \psi$', ha='center', va='bottom', fontsize = 9)
        fig_kh.text(0.01, 0.5, r'$\Delta \psi$', ha='left', va='center', rotation='vertical', fontsize=9)
        ax_kh[-1].set_xlim(pilims)
        ax_kh[-1].set_ylim(pilims)
        fig_kh.subplots_adjust(hspace=0.015, wspace=0.015,left=0.10, bottom=0.10,top=0.95, right=0.95)
        fig_kh.subplots_adjust(hspace=0.00, wspace=0.00,left=0.07, bottom=0.06,top=0.95, right=0.97)
        #fig_kh.tight_layout()
        if pub_fig:
            fig_kh.savefig(filename, bbox_inches='tight', pad_inches=0.01)

        fig_kh.suptitle(suptitle,fontsize=8)
        fig_kh.canvas.draw(); fig_kh.show()
        return fig_kh, ax_kh

    def plot_clusters_phase_lines(self,decimation=4000, linewidth=0.05, colours = colours1,xlabel_loc=0.5, ylabel_loc=3.2, yline=0,xlabel=''):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line
        if decimation > 2000, it is the number of points desired
        set yline to draw a constant y line for reference (or None)

        SH: 9May2013
        '''
        # autodecimation has problems with fewer than 5000 points
        npts = len(self.cluster_assignments)
        n1 = npts//max(npts//decimation,1) # assume decimation is desired no
        n2 = npts//decimation             # assume it is the reduction factor
        #print(n1,n2)

        decimation = max(1,npts//max(n1, n2))

        if npts//decimation < 500:
            print('decimation of {d} is probably too high for {n} points'.
                  format(d=decimation, n=npts))

        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)<=10:  # don't bother with small ones
                print('omitting cluster {c}'.format(c=c))
            else:
                tmp = self.feature_obj.instance_array[current_items,:]%(2.*np.pi)
                tmp[tmp>np.pi]-=(2.*np.pi)
                pcent = str('{cl}: {pc:.1f}%'
                            .format(cl=cluster, 
                                    pc=100*np.sum(current_items)/float(npts)))
                if yline is not None:
                    ax[cluster].plot([0,len(tmp[0])], [yline,yline],
                                     'gray',linewidth=0.5)
                ax[cluster].plot(tmp[::decimation,:].T,'-'
                                 ,color = colours[cluster % len(colours)]
                                 , linewidth=linewidth)
                ax[cluster].text(xlabel_loc,ylabel_loc, pcent, 
                                 horizontalalignment='left',
                                 verticalalignment='bottom',
                                 bbox=dict(facecolor='lightgray',
                                           alpha=0.4,color='gray'))
                # ax[cluster].legend(loc='best')  # get too many

        bottom = 0.05
        if xlabel != '': #ax[cluster].set_xlabel(xlabel)
            fig.text(0.5, 0.01,xlabel)
            bottom += .015


        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=bottom ,top=0.95, right=0.95)
        fig.suptitle(suptitle, fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax

##INS :    def plot_clusters_polarisations(self, decimati
## BEWARE - probably ignores angle calibration! must update with https://github.com/shaunhaskey/pyfusion/blob/SH_branch/pyfusion/clustering/clustering.py#L1109
    def plot_clusters_polarisations(self, 
                                    coil_numbers=None, decimate=None, polar_plot=None, y_axis=None, pub_fig=None, fig_name=None, inc_title=None, energy=None, plot_amps=None, plot_distance=None, angle_error=None,
                                    decimation=1, single_plot = False, kappa_cutoff = None, cumul_sum = False, cluster_list = None, ax = None, colours = None, scatter_kwargs = None):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        
        if coil_numbers is not None:
            print('coil_numbers guessed')
            coil_numbers = range(16)
        if decimate is not None:
            print('decimation assumed')
            decimation = decimate

        if scatter_kwargs == None: scatter_kwargs = {'s':100, 'alpha':0.05,'linewidth':'1'}
        ax_supplied = False if ax==None else True
        cluster_list_tmp = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list_tmp)
        if cluster_list==None:  cluster_list = cluster_list_tmp
        debug_(pyfusion.DEBUG, 2, key='enter polarisation')
        if single_plot:
            if not ax_supplied: fig, ax = pt.subplots(); ax = [ax]*n_clusters
            if colours == None: colours = ['r','k','b','y','m']*10
        else:
            if not ax_supplied: fig, ax = make_grid_subplots(n_clusters, sharex = True, sharey = True)
            if colours == None: colours = ['k']*n_clusters
        if kappa_cutoff!=None:
            averages = np.average(self.cluster_details["EM_VMM_kappas"],axis=1)
            cluster_list = np.arange(len(averages))[averages>kappa_cutoff]
        marker_list = ['o' for i in colours]
        means = []
        count = 0
        axes_list = []
        for cluster in cluster_list:
            print('cluster', str(cluster))
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                print('current_items', len(current_items))
                coil_col_array = int(np.sum(current_items))*[3*coil_numbers]
                coil_num_array = int(np.sum(current_items))*[[i for i in coil_numbers for j in [0,1,2]]]
                tmp = np.abs(self.feature_obj.misc_data_dict['naked_coil'][current_items,:])
                tmp2 = self.feature_obj.misc_data_dict['freq'][current_items]
                tmp /= np.sqrt(np.sum(tmp**2, axis = 1))[:,np.newaxis]
                plot_ax = ax[cluster] if not ax_supplied else ax[count]
                # ALL 3 CPTS COLOURED SPEARATELY
                plot_ax.scatter(coil_num_array, np.sqrt(tmp[:]**2), c=np.array(coil_col_array)%3/4., marker=marker_list[count], cmap=None, norm=None, zorder=0, rasterized=True)
                #plot_ax.scatter(tmp[:,1], np.sqrt(tmp[:,0]**2 + tmp[:,2]**2), c=colours[count], marker=marker_list[count], cmap=None, norm=None, zorder=0, rasterized=True)
                #plot_ax.scatter(tmp[:,0], tmp2, c=colours[count], marker=marker_list[count], cmap=None, norm=None, zorder=0, rasterized=True,alpha = 0.1)
                # two cpts "polar style"
                plot_ax.scatter(tmp[:,1], tmp[:,0], c=colours[count], marker=marker_list[count], cmap=None, norm=None, zorder=0, rasterized=True)
                print(np.mean(np.sum(tmp**2, axis = 1)))
                axes_list.append(plot_ax)
                count+=1
        ax[0].set_xlim([0,self.cluster_details['EM_VMM_means'].shape[1]])
        #if not cumul_sum: ax[0].set_ylim([0, 1]); ax[0].set_xlim([0,1])
        debug_(pyfusion.DEBUG, 1, key='mid polarisation')
        for i in ax:i.grid(True)
        if not ax_supplied:
            fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig.suptitle(suptitle.replace('_','-'), fontsize = 8)
            fig.canvas.draw(); fig.show()
            return fig, ax
        

    def plot_clusters_amp_lines(self,decimation=1, linewidth=0.05):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                #tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]).T / np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])).T
                tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,1:]) / np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0:-1]))
                ax[cluster].plot(tmp[::decimation,:].T,'k-',linewidth=linewidth)
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle, fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_fft_amp_lines(self,decimation=1):
        '''Plot all the phase lines for the clusters
        Good clusters will show up as dense areas of line

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                #tmp = (np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:]).T / np.abs(self.feature_obj.misc_data_dict['mirnov_data'][current_items,0])).T
                tmp = np.fft.fft(self.feature_obj.misc_data_dict['mirnov_data'][current_items,:])
                print(np.max(np.abs(tmp),axis=1).shape)
                tmp = ((np.abs(tmp).T)/np.max(np.abs(tmp),axis=1)).T
                ax[cluster].plot(np.abs(tmp[::decimation,:]).T,'k-',linewidth=0.05)
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle, fontsize = 8)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_interferometer_channels(self, interferometer_spacing=0.025, interferometer_start=0,  include_both_sides = 1, plot_phases=0):
        '''plot kh vs frequency for each cluster - i.e looking for whale tails
        The colouring of the points is based on the total phase along the array
        i.e a 1D indication of the clusters

        SH: 9May2013
        '''
        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        for i in ax: i.set_rasterization_zorder(1)
        if plot_phases : fig_phase, ax_phase = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        passed = True; i=1; ne_omega_data = []
        misc_data_dict = self.feature_obj.misc_data_dict
        #strange way to determine the number of channels we have.... need to figureout a better way
        ne_omega_data = misc_data_dict['ne_mode']

        #this is a bad fudge....
        #channel_list = np.arange(interferometer_start,interferometer_spacing*len(ne_omega_data),interferometer_spacing)
        channel_list = np.arange(interferometer_start,interferometer_spacing*ne_omega_data.shape[1],interferometer_spacing)
        #ne_omega_data = np.array(ne_omega_data).transpose()
        for cluster in cluster_list:
            current_items = self.cluster_assignments==cluster
            if np.sum(current_items)>10:
                current_data = np.abs(ne_omega_data[current_items,:])
                rms_amp = np.sqrt(np.sum((current_data * current_data),axis=1))
                mean_rms = np.mean(rms_amp)
                large_enough = rms_amp >mean_rms
                current_data = current_data / np.tile(rms_amp,(7,1)).transpose()
                if plot_phases:  
                    current_phase = np.angle(ne_omega_data[current_items,:])
                    current_phase = current_phase - np.tile(current_phase[:,2],(7,1)).transpose()
                    current_phase = np.rad2deg(current_phase %(2.*np.pi))
                ax[cluster].plot(channel_list, current_data[large_enough].T,'k-',linewidth=0.02,zorder = 0)
                means = np.mean(current_data[large_enough], axis=0)
                std = np.std(current_data[large_enough], axis=0)
                ax[cluster].errorbar(channel_list, means, yerr=std)
                ax[cluster].plot(channel_list, means,'c-o',linewidth=6)
                ax[cluster].plot(channel_list, means,'b-o',linewidth=4)
                ax[cluster].plot(channel_list*-1, means,'c--o',linewidth=6)
                ax[cluster].plot(channel_list*-1, means,'b--o',linewidth=4)
                ax[cluster].grid()
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        if plot_phases:
            fig_phase.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
            fig_phase.suptitle(suptitle,fontsize=8)
            fig_phase.canvas.draw(); fig_phase.show()
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_single_kh(self, cluster_list = None,kappa_cutoff=None,color_by_cumul_phase = 1, sqrtne=None, plot_alfven_lines=1,xlim=None,ylim=None,pub_fig = 0, filename = None, marker_size = 100):
        '''plot kh vs frequency for each cluster - i.e looking for whale tails
        The colouring of the points is based on the total phase along the array
        i.e a 1D indication of the clusters

        can provide a cluster_list to select which ones are plotted
        or can give a kappa_cutoff (takes precedent)
        otherwise, all will be plotted
        SH: 9May2013
        '''
        if pub_fig:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.5
            mpl.rcParams['savefig.dpi']=100

        misc_data_dict = self.feature_obj.misc_data_dict
        if kappa_cutoff!=None:
            averages = np.average(self.cluster_details["EM_VMM_kappas"],axis=1)
            fig, ax = pt.subplots()
            kappa_cutoff_list = range(50)
            items = []
            for kappa_cutoff_tmp in kappa_cutoff_list:
                cluster_list = np.arange(len(averages))[averages>kappa_cutoff_tmp]
                total = 0
                for i in cluster_list: total+= np.sum(self.cluster_assignments==i)
                items.append(total)
            ax.plot(kappa_cutoff_list, items,'o')
            fig.canvas.draw(); fig.show()
            cluster_list = np.arange(len(averages))[averages>kappa_cutoff]
            total = 0
            for i in cluster_list: total+= np.sum(self.cluster_assignments==i)
            print('total clusters satisfying kappa bar>{}:{}'.format(kappa_cutoff,total))

        elif cluster_list  is None:
            cluster_list = list(set(self.cluster_assignments))
        kh_plot_item = 'kh'; freq_plot_item = 'freq'
        fig, ax = pt.subplots(); ax = [ax]
        if pub_fig:
            fig.set_figwidth(8.48*cm_to_inch)
            fig.set_figheight(8.48*0.8*cm_to_inch)
        for i in ax: i.set_rasterization_zorder(1)
        if color_by_cumul_phase:
            instance_array2 = modtwopi(self.feature_obj.instance_array, offset=0)
            max_lim = -1*2.*np.pi; min_lim = (-3.*2.*np.pi)
            total_phase = np.sum(instance_array2,axis=1)
            total_phase = np.clip(total_phase,min_lim, max_lim)
        plotting_offset = 0
        colour_list = ['k','b','r','y','g','c','m','w']
        marker_list = ['o' for i in colour_list]
        marker_list.extend(['d' for i in colour_list])
        colour_list.extend(colour_list)
        print('hello')
        while len(colour_list)< len(cluster_list):
            print('hello')
            colour_list.extend(colour_list)
            marker_list.extend(marker_list)
        for i,cluster in enumerate(cluster_list):
            current_items = self.cluster_assignments==cluster
            if sqrtne is None:
                scatter_data = misc_data_dict[freq_plot_item][current_items]/1000
            else:
                scatter_data = misc_data_dict[freq_plot_item][current_items]*np.sqrt(misc_data_dict['ne{ne}'.format(ne=sqrtne)][current_items])
                print('scaling by ne')
            if np.sum(current_items)>10:
                if color_by_cumul_phase == 1:
                    print('hello instance', cluster)
                    ax[0].scatter((misc_data_dict[kh_plot_item][current_items]), scatter_data, s=marker_size, c=total_phase[current_items], vmin = min_lim, vmax = max_lim, marker='o', cmap='jet', norm=None, alpha=0.05)
                elif color_by_cumul_phase ==0:
                    ax[0].scatter((misc_data_dict[kh_plot_item][current_items]), scatter_data,s=marker_size, c=colour_list[i], marker=marker_list[i], cmap=None, norm=None, alpha=0.05,zorder=0,rasterized=True)
                    print('hello, no instance', cluster)
                elif color_by_cumul_phase == 2:
                    ax[0].scatter((misc_data_dict[kh_plot_item][current_items]), scatter_data,s=marker_size, c='k', marker='o', cmap=None, norm=None, alpha=0.05,zorder=0,rasterized=True)
                    print('hello, no instance', cluster)
        if plot_alfven_lines:
            plot_alfven_lines_func(ax[0])
        ax[-1].set_xlim([0.201,0.99])
        ax[0].set_xlabel(r'$\kappa_H$')
        #ax[0].set_ylabel(r'$\omega \sqrt{n_e}$')
        ax[0].set_ylabel('Frequency (kHz)')
        if xlim!=None: ax[0].set_xlim(xlim)
        if ylim!=None: ax[0].set_ylim(ylim)
        if pub_fig:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.01)

        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw(); fig.show()
        return fig, ax

    def plot_cumulative_phases(self,):
        if ((self.settings['method']) == ('EM_VMM')) or ((self.settings['method']) == ('EM_VMM_soft')): 
            means = self.cluster_details['EM_VMM_means']
        elif self.settings['method'] == 'k_means':
            means = self.cluster_details['k_means_centroids']
        elif self.settings['method'] == 'k_means_periodic':
            means = self.cluster_details['k_means_periodic_centroids']

        cluster_list = list(set(self.cluster_assignments))
        suptitle = self.settings.__str__().replace("'",'').replace("{",'').replace("}",'')
        n_clusters = len(cluster_list)
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        for cluster in cluster_list:
            cluster_phases = means[cluster][:]
            cumulative_phase = [0]
            for tmp_angle in range(len(cluster_phases)):
                cumulative_phase.append(cumulative_phase[-1] + cluster_phases[tmp_angle])
            cumulative_phase = np.array(cumulative_phase)/(2.*np.pi)
            ax[cluster].plot(range(len(cumulative_phase)),cumulative_phase,'o-')
            colors = ['k','b','r']; plot_style = ['-o','-x','-s']; plot_style2 = ['--o','--x','--s']

            #approximate coil locations in Boozer Coordinates
            min_locations_theta = np.array([297.165,268.283,239.198,211.107,185.257,160.934,
                                            137.809,114.798,92.381,70.123,46.695,21.438,-5.049,
                                            -32.694,-61.366,-90.180])
            min_locations_phi = np.array([46.742,37.825,28.680,19.329,9.922,0.604,-8.267,-16.854,
                                          -24.904,-32.505,-40.037,-47.792,-55.819,-64.136,
                                          -72.807,-81.693])
            #Boozer mode list
            m_mode_list = [3,4,5]; n_mode_list = [-4,-5,-6]
            for j in range(0,len(m_mode_list)):
                m_mode = m_mode_list[j]; n_mode = n_mode_list[j]
                phases = (m_mode * min_locations_theta[1:]/180.*np.pi + n_mode * min_locations_phi[1:]/180.*np.pi)
                diff_phases = np.diff(phases)
                min_amount = -1.5
                diff_phases[diff_phases<min_amount*np.pi]+=2.*np.pi
                diff_phases[diff_phases>(2+min_amount)*np.pi]-=2.*np.pi
                cumulative_phases = (np.cumsum(diff_phases))/2./np.pi
                cumulative_phases = np.append([0],cumulative_phases)
                ax[cluster].plot(cumulative_phases,colors[j],label='(%d,%d)'%(n_mode,m_mode))

        ax[-1].set_ylim([-3,0])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle,fontsize=8)

        leg = ax[-1].legend(prop={'size':8.0})
        leg.get_frame().set_alpha(0.5)

        fig.canvas.draw(); fig.show()

    def plot_EMM_GMM_amps(self,suptitle = ''):
        fig, ax = make_grid_subplots(self.settings['n_clusters'])
        means = self.cluster_details['EM_GMM_means']
        stds = self.cluster_details['EM_GMM_std']
        for i in range(means.shape[0]):
            ax[i].plot(means[i,:])
            ax[i].plot(means[i,:] + stds[i,:])
            ax[i].plot(means[i,:] - stds[i,:])
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.suptitle(suptitle,fontsize=8)
        fig.canvas.draw(); fig.show()


    def cluster_probabilities(self,):
        tmp = np.max(self.cluster_details['zij'],axis=1)
        tmp1 = np.sum(self.cluster_details['zij'],axis=1)
        print('best prob: {best_prob:.3f}, worst_prob: {worst_prob:.3f}, max row sum: {max_row:.2f}, min row sum: {min_row:.2f}'.format(best_prob = np.max(tmp), worst_prob=np.min(tmp), max_row=np.max(tmp1), min_row=np.min(tmp1)))
        n_clusters = len(list(set(self.cluster_assignments)))
        fig, ax = make_grid_subplots(n_clusters, sharex = 'all', sharey = 'all')
        for i in list(set(self.cluster_assignments)):
            curr_probs = tmp[self.cluster_assignments==i]
            print('cluster {clust}, min prob {min:.2f}, max prob {max:.2f}, mean prob {mean:.2f}, std dev {std:.2f}'.format(clust = i, min = np.min(curr_probs), max = np.max(curr_probs), mean = np.mean(curr_probs), std= np.std(curr_probs)))
            ax[i].hist(curr_probs, bins=300)
        fig.canvas.draw(); fig.show()


class clusterer_wrapper(clustering_object):
    '''Wrapper around the EM_GMM_clustering function
    Decided to use a wrapper so that it can be used outside of this architecture if needed

    method : k-means, EMM_GMM, k_means_periodic, EM_VMM
    pass settings as kwargs: these are the default settings:
    'k_means': {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30,'seed':1,'use_scikit':1}
    'EM_GMM' : {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30},
    'k_means_periodic' : {'n_clusters':9, 'number_of_starts':10, 'n_cpus':1, 'distance_calc':'euclidean','convergence_diff_cutoff': 0.2, 'iterations': 40, 'decimal_roundoff':2},
    'EM_VMM' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1}}
    'EM_GMM2' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1}}

    SH: 6May2013
    '''
    def __init__(self, feature_obj, method='k-means', **kwargs):
        self.feature_obj = feature_obj
        print('kwargs', kwargs)

        #Default settings are declared first, which are overwritten by kwargs
        #if appropriate- this is for record keeping of all settings that are used
        cluster_funcs = {'k_means': k_means_clustering, 'EM_GMM' : EM_GMM_clustering,
                         'k_means_periodic' : k_means_periodic, 'EM_VMM' : EM_VMM_clustering_wrapper,
                         'EM_GMM2' : EM_GMM_clustering_wrapper,
                         'EM_VMM_GMM': EM_VMM_GMM_clustering_wrapper}

        #EM_VMM_clustering,'EM_VMM_soft':EM_VMM_clustering_soft}
        cluster_func_class = {'k_means': 'func', 'EM_GMM' : 'func',
                         'k_means_periodic' : 'func', 'EM_VMM' : 'func', 'EM_GMM2':'func', 'EM_VMM_GMM':'func'}
        
        default_settings = {'k_means': {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30,'seed':1,'use_scikit':1},
                            'EM_GMM' : {'n_clusters':9, 'sin_cos':1, 'number_of_starts':30},
                            'k_means_periodic' : {'n_clusters':9, 'number_of_starts':10, 'n_cpus':1, 'distance_calc':'euclidean','convergence_diff_cutoff': 0.2, 'n_iterations': 40, 'decimal_roundoff':2},
                            'EM_VMM' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                                        'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                        'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1},
                            'EM_GMM2' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'k_means',
                                         'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                         'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1},
                            'EM_VMM_GMM' : {'n_clusters':9, 'n_iterations':20, 'n_cpus':1, 'start':'random',
                                        'kappa_calc':'approx','hard_assignments':0,'kappa_converged':0.2,
                                        'mu_converged':0.01,'LL_converged':1.e-4,'min_iterations':10,'verbose':1,'number_of_starts':1}}

        #EM_VMM_GMM_clustering_wrapper(instance_array, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):
        #replace EM_VMM and EM_VMM_soft with the class.... somehow
        self.settings = default_settings[method]
        self.settings.update(kwargs)
        cluster_func = cluster_funcs[method]
        print(method, self.settings)
        if cluster_func_class[method]=='func':
            print('func based...')
            if method!='EM_VMM_GMM':
                self.cluster_assignments, self.cluster_details = cluster_func(self.feature_obj.instance_array, **self.settings)
            else:
                self.cluster_assignments, self.cluster_details = cluster_func(self.feature_obj.instance_array, self.feature_obj.misc_data_dict['mirnov_data'], **self.settings)
        else:
            print('class based...')
            tmp = cluster_func(self.feature_obj.instance_array, **self.settings)
            self.cluster_assignments, self.cluster_details = tmp.cluster_assignments, tmp.cluster_details
        self.settings['method']=method
        #self.cluster_details['header']='testing'

##INS  1194:def normalise_covariances(cov_mat, geom = True):
##INS   1206:def pearson_covariances(cov_mat):

###############################################################
def show_covariances(gmm_covars_tmp, clim=None,individual=None,fig_name=None):
    fig, ax = make_grid_subplots(gmm_covars_tmp.shape[0], sharex = 'all', sharey = 'all')
    im = []
    for i in range(gmm_covars_tmp.shape[0]):
        im.append(ax[i].imshow(np.abs(gmm_covars_tmp[i,:,:]),aspect='auto', interpolation='nearest'))
        print(im[-1].get_clim())
        if clim is None:
            im[-1].set_clim([0, im[-1].get_clim()[1]*0.5])
        else:
            im[-1].set_clim(clim)
    if individual!=None:
        fig_ind,ax_ind = pt.subplots(nrows=len(individual),sharex = 'all',sharey = 'all')
        if fig_name!=None:
            cm_to_inch=0.393701
            import matplotlib as mpl
            old_rc_Params = mpl.rcParams
            mpl.rcParams['font.size']=8.0
            mpl.rcParams['axes.titlesize']=8.0#'medium'
            mpl.rcParams['xtick.labelsize']=8.0
            mpl.rcParams['ytick.labelsize']=8.0
            mpl.rcParams['lines.markersize']=1.0
            mpl.rcParams['savefig.dpi']=300
            fig_ind.set_figwidth(8.48*cm_to_inch)
            fig_ind.set_figheight(8.48*0.8*cm_to_inch)

        if len(individual)==1:ax_ind = [ax_ind]
        for i,clust in enumerate(individual):
            im = ax_ind[i].imshow(np.abs(gmm_covars_tmp[clust,:,:]),aspect='auto', interpolation='nearest')
            cbar = pt.colorbar(im,ax=ax_ind[i])
            cbar.set_label('covariance')
            im.set_clim(clim)
            ax_ind[i].set_ylabel('Channel')
        ax_ind[-1].set_xlabel('Channel')
        if fig_name!=None:
            fig_ind.savefig(fig_name, bbox_inches='tight', pad_inches=0.01)
        fig_ind.canvas.draw();fig_ind.show()
    #for i in im : i.set_clim(clims)
    fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
    fig.canvas.draw();fig.show()


##################Clustering wrappers#########################
def EM_GMM_clustering(instance_array, n_clusters=9, sin_cos = 0, number_of_starts = 10, show_covariances = 0, clim=None, covariance_type='diag'):
    print('starting EM-GMM algorithm from sckit-learn, k=%d, retries : %d, sin_cos = %d'%(n_clusters,number_of_starts,sin_cos))
    if sin_cos==1:
        print('  using sine and cosine of the phases')
        sin_cos_instances = np.zeros((instance_array.shape[0],instance_array.shape[1]*2),dtype=float)
        sin_cos_instances[:,::2]=np.cos(instance_array)
        sin_cos_instances[:,1::2]=np.sin(instance_array)
        input_data = sin_cos_instances
    else:
        print('  using raw phases')
        input_data = instance_array
    gmm = mixture.GMM(n_components=n_clusters,covariance_type=covariance_type,n_init=number_of_starts)
    gmm.fit(input_data)
    cluster_assignments = gmm.predict(input_data)
    bic_value = gmm.bic(input_data)
    LL = np.sum(gmm.score(input_data))
    gmm_covars_tmp = np.array(gmm._get_covars())
    if show_covariances:
        fig, ax = make_grid_subplots(gmm_covars_tmp.shape[0], sharex = 'all', sharey = 'all')
        im = []
        for i in range(gmm_covars_tmp.shape[0]):
            im.append(ax[i].imshow(np.abs(gmm_covars_tmp[i,:,:]),aspect='auto'))
            print(im[-1].get_clim())
            if clim is None:
                im[-1].set_clim([0, im[-1].get_clim()[1]*0.5])
            else:
                im[-1].set_clim(clim)
        clims = [np.min(np.abs(gmm_covars_tmp)),np.max(np.abs(gmm_covars_tmp))*0.5]
        #for i in im : i.set_clim(clims)
        fig.subplots_adjust(hspace=0, wspace=0,left=0.05, bottom=0.05,top=0.95, right=0.95)
        fig.canvas.draw();fig.show()

    gmm_covars = np.array([np.diagonal(i) for i in gmm._get_covars()])
    gmm_means = gmm.means_
    if sin_cos:
        cluster_details = {'EM_GMM_means_sc':gmm_means, 'EM_GMM_variances_sc':gmm_covars, 'EM_GMM_covariances_sc':gmm_covars_tmp,'BIC':bic_value, 'LL':LL}
    else:
        cluster_details = {'EM_GMM_means':gmm_means, 'EM_GMM_variances':gmm_covars, 'EM_GMM_covariances':gmm_covars_tmp, 'BIC':bic_value,'LL':LL}
    return cluster_assignments, cluster_details

def k_means_clustering(instance_array, n_clusters=9, sin_cos = 1, number_of_starts = 30, seed=None,use_scikit=1,**kwargs):
    '''
    This runs the k-means clustering algorithm as implemented in scipy - change to scikit-learn?

    SH: 7May2013
    '''
    from sklearn.cluster import KMeans
    print('starting kmeans algorithm, k=%d, retries : %d, sin_cos = %d'%(n_clusters,number_of_starts,sin_cos))
    if sin_cos==1:
        print('  using sine and cosine of the phases')
        sin_cos_instances = np.zeros((instance_array.shape[0],instance_array.shape[1]*2),dtype=float)
        sin_cos_instances[:,::2]=np.cos(instance_array)
        sin_cos_instances[:,1::2]=np.sin(instance_array)
        input_array = sin_cos_instances
        #code_book,distortion = vq.kmeans(sin_cos_instances, n_clusters,iter=number_of_starts)
        #cluster_assignments, point_distances = vq.vq(sin_cos_instances, code_book)
    else:
        print('  using raw phases')
        input_array = instance_array
        #code_book,distortion = vq.kmeans(instance_array, n_clusters,iter=number_of_starts)
        #cluster_assignments, point_distances = vq.vq(instance_array, code_book)
    #pickle.dump(multiple_run_results,file(k_means_output_filename,'w'))
    if use_scikit:
        print('using scikit learn')
        tmp = KMeans(init='k-means++', n_clusters=n_clusters, n_init = number_of_starts, n_jobs=1, random_state = seed)
        cluster_assignments = tmp.fit_predict(input_array)
        code_book = tmp.cluster_centers_
    else:
        print('using vq from scipy')
        code_book,distortion = vq.kmeans(input_array, n_clusters,iter=number_of_starts)
        cluster_assignments, point_distances = vq.vq(input_array, code_book)
    if sin_cos:
        cluster_details = {'k_means_centroids_sc':code_book}
    else:
        cluster_details = {'k_means_centroids':code_book}
    return cluster_assignments, cluster_details

##################################################################################
#############################k-means periodic algorithm##############################
def new_method(tmp_array):
    #find the items above and below np.pi as this will determine their wrapping behaviour
    gt_pi = tmp_array>=np.pi
    less_pi = tmp_array<np.pi
    items = tmp_array.shape[0]

    #find the breakpoints for wrapping
    break_points = copy.deepcopy(tmp_array)
    break_points[less_pi] += np.pi
    break_points[gt_pi] -= np.pi

    #create a list of unique subintervals in order and append 2pi to it
    subintervals = np.append(np.unique(break_points), 2.*np.pi)
    wk = []; q = []

    #total sum of the array - this will be modified by factors of 2pi depending on which
    #sub interval we are in
    total_sum = np.sum(tmp_array)
    for i in range(0,subintervals.shape[0]):
        tmp_points2 = (break_points<=subintervals[i])*(less_pi)
        tmp_points3 = (break_points>subintervals[i])*(gt_pi)
        #tmp_array2[tmp_points2] += 2.*np.pi
        
        #calculate new centroid and make sure it lies in <0, 2pi)
        wk.append(((total_sum + (np.sum(tmp_points2)- np.sum(tmp_points3))*2.*np.pi)/items)%(2.*np.pi))
        #old way - not possible mistake with (wk[-1] < 2.*np.pi) - should be <0???
        #wk.append((total_sum + (np.sum(tmp_points2)- np.sum(tmp_points3))*2.*np.pi)/items)
        # while (wk[-1] >= 2.*np.pi) or (wk[-1]< 0):
        #     if (wk[-1] >= 2.*np.pi):
        #         wk[-1] -= 2.*np.pi
        #     elif (wk[-1] < 2.*np.pi):
        #         wk[-1] += 2.*np.pi
        tmp = np.abs(tmp_array - wk[-1])
        tmp = np.minimum(tmp, 2.*np.pi-tmp)
        q.append(0.5*np.sum(tmp**2))
    return wk[np.argmin(q)], np.min(q)

def _k_means_p_centroid_1d(tmp_array):
    '''Function to find the cluster centroid in one dimension
    around a circle

    Note this has become fairly highly vectorised and is the most
    time consuming part of the calcuation

    SH: 9May2013
    '''
    #find the items above and below np.pi as this will determine their wrapping behaviour
    tmp_array = np.sort(tmp_array)
    items = len(tmp_array)
    subintervals= np.unique(tmp_array)
    n_subintervals = len(subintervals)
    subinterval_locs = tmp_array.searchsorted(subintervals)
    items_per_subinterval = np.diff(subinterval_locs)
    items_per_subinterval = np.append(items_per_subinterval, items - np.sum(items_per_subinterval))
    #total_sum = np.sum(tmp_array)
    total_sum = np.sum(items_per_subinterval*subintervals) #marginally faster
    gt_pi_index = subintervals.searchsorted(np.pi)

    #deal with subintervals from 0->pi
    #find the breakpoints for wrapping
    break_points = np.zeros(n_subintervals,dtype=np.float)
    break_points[0:gt_pi_index] = subintervals[0:gt_pi_index]+np.pi
    break_points[gt_pi_index:] = subintervals[gt_pi_index:]-np.pi
    points_to_end = np.cumsum(items_per_subinterval[::-1])
    points_from_start = np.cumsum(items_per_subinterval)

    correction = np.zeros(n_subintervals,dtype=np.float)
    indices = np.minimum(subintervals.searchsorted(break_points[0:gt_pi_index]),n_subintervals-1)
    correction[:gt_pi_index] = points_to_end[indices]*(-2.*np.pi)

    indices = subintervals.searchsorted(break_points[gt_pi_index:])
    correction[gt_pi_index:] = points_from_start[indices]*(2.*np.pi)

    averages = ((total_sum + correction) / items)%(2.*np.pi)

    sub_ints1 = np.tile(subintervals,(n_subintervals,1))
    tmp = np.abs(sub_ints1 - averages[:,np.newaxis])

    tmp = np.minimum(tmp,2.*np.pi-tmp) * items_per_subinterval[:,np.newaxis]
    q_vals = np.sum(tmp**2,axis=1)
    return averages[np.argmin(q_vals)], np.min(q_vals)



def _k_means_p_centroid_1d_complex_average(tmp_array):
    '''Function to find the cluster centroid in one dimension
    around a circle - using the average of the complex numbers...
    this is a biased estimator....

    SH: 9May2013
    '''
    #find the items above and below np.pi as this will determine their wrapping behaviour
    c = np.mean(np.cos(tmp_array))
    s = np.mean(np.sin(tmp_array))
    mean = np.arctan2(s,c)
    distances = np.abs(tmp_array - mean)
    qvals = np.sum((np.minimum(distances, 2.*np.pi-distances))**2)
    return mean, qvals
    #sub_ints1 = np.tile(subintervals,(n_subintervals,1))
    #tmp = np.abs(sub_ints1 - averages[:,np.newaxis])

    # tmp = np.minimum(tmp,2.*np.pi-tmp) * items_per_subinterval[:,np.newaxis]
    # q_vals = np.sum(tmp**2,axis=1)
    # return averages[np.argmin(q_vals)], np.min(q_vals)



def _k_means_p_calc_centroids(centroid, cluster_assignments, instance_array, k):
    '''Find all the new centroids in all dimensions for all clusters
    uses _k_means_p_centroid_1d to perform the calculation

    SH: 9May2013
    '''
    q_tot = 0
    for i in range(0,k):
        relevant_indices = (cluster_assignments == i)
        #ignore clusters without any members
        if instance_array[relevant_indices,:].shape[0] == 0:
            pass
        else:
            #extract the members of the cluster
            tmp_array = instance_array[relevant_indices,:]
            #treat each attribute seperately, and add up the q values
            for j in range(0,tmp_array.shape[1]):
                #centroid[i,j], q_val = _k_means_p_centroid_1d(tmp_array[:,j])
                centroid[i,j], q_val = _k_means_p_centroid_1d_complex_average(tmp_array[:,j])
                q_tot += q_val
    return centroid, q_tot

def _k_means_p_calc_distance(instance_array, centroids, distance_calc = 'euclidean'):
    '''Calculate the distances for all points to all clusters
    this is used to assign the points to clusters 

    SH:9May2013
    '''
    distances = np.ones((instance_array.shape[0],centroids.shape[0]),dtype=np.float)
    for i in range(0, centroids.shape[0]):
        tmp = np.abs(instance_array - np.tile(centroids[i,:],(instance_array.shape[0],1)))
        tmp = np.minimum(tmp, 2.*np.pi-tmp)
        if distance_calc == 'manhatten':
            tmp = np.sum(np.abs(tmp),axis = 1)
        else:
            tmp = np.sum(np.abs(tmp*tmp),axis = 1)
        distances[:,i] = copy.deepcopy(tmp)
    cluster_assignments = np.argmin(distances,axis = 1)
    return cluster_assignments

def _k_means_p_rand_centroids(k, d, seed):
    '''
    Create the random centroid array to start the k-means algorithm 
    based on a seed so that it is repeatable.

    SH: 8May2013
    '''
    np.random.seed(seed)
    centroids = (2.*np.pi)*np.random.random((k,d))#-np.pi
    return centroids

def _k_means_p_single_seed(k, seed, instance_array, distance_calc, convergence_diff_cutoff, iterations):
    '''
    This is the function that calculates the k-means periodic clustering

    SH: 8May2013
    '''
    centroids = _k_means_p_rand_centroids(k, instance_array.shape[1], seed)
    old_centroid = copy.copy(centroids)
    convergence = []; q_list = []
    current_iteration = 0; curr_diff = np.inf
    while (curr_diff > convergence_diff_cutoff) and (current_iteration < iterations):
        current_iteration += 1
        start_time = time.time()
        cluster_assignments = _k_means_p_calc_distance(instance_array, centroids, distance_calc = distance_calc)
        distance_time = time.time()
        new_centroids, q_curr = _k_means_p_calc_centroids(copy.copy(centroids), cluster_assignments, instance_array, k)
        centroid_time = time.time()
        q_list.append(q_curr)
        convergence.append(np.sum(np.abs(new_centroids - centroids)))
        centroids = copy.copy(new_centroids)
        if current_iteration >= 2:
            curr_diff = np.abs(q_list[-1] - q_list[-2])
        print('pid : %d, iteration : %3d, convergence : %10.3f, q_diff : %10.4f, q_tot : %10.2f, times : %5.3fs %.3fs %.3fs'%(os.getpid(), current_iteration, convergence[-1], curr_diff, q_list[-1], distance_time-start_time, centroid_time-distance_time, time.time() - start_time))
    return centroids, cluster_assignments, q_list[-1]

def _k_means_p_multiproc_wrapper(arguments):
    '''Wrapper around _k_means_p_single_seed so that it can be used with
    multiprocessing, and be passed multiple arguments

    SH: 8May2013
    '''
    print('started wrapper')
    return _k_means_p_single_seed(*arguments)

def k_means_periodic(instance_array, n_clusters = 9, number_of_starts = 10, n_cpus=1, distance_calc = 'euclidean',convergence_diff_cutoff = 0.2, n_iterations = 40, decimal_roundoff=2,seed_list=None, **kwargs):
    k = n_clusters
    #round, take relevant columns and make sure all datapoints are on the interval <0,2pi)
    instance_array = copy.deepcopy(instance_array)
    instance_array = instance_array.astype(np.float)
    #take modulus 2pi
    instance_array = instance_array % (2.*np.pi)
    #round to decimal roundoff places
    instance_array = np.round(np.array(instance_array),decimals = decimal_roundoff)
    #ensure that the instances are are still [0,2pi)
    instance_array = instance_array % (2.*np.pi)
    print(np.max(instance_array)>(2.*np.pi), np.min(instance_array)<0)
    #prepare seeds if they weren't provided
    if seed_list is None:
        seed_list = map(int, np.round(np.random.rand(number_of_starts)*100.))
    multiple_run_results = {}; q_val_list = []
    if n_cpus>1:
        pool_size = n_cpus
        print('  pool size :', pool_size)
        pool = multiprocessing.Pool(processes=pool_size)
        results = pool.map(_k_means_p_multiproc_wrapper, 
                           izip(itertools.repeat(k), seed_list, itertools.repeat(instance_array),
                                          itertools.repeat(distance_calc), itertools.repeat(convergence_diff_cutoff),
                                          itertools.repeat(n_iterations)))
        print('  closing pool and waiting for pool to finish')
        pool.close(); pool.join() # no more tasks
        print('  pool finished')
    else:
        results = []
        for seed in seed_list:
            multiple_run_results[seed] = {}
            results.append(_k_means_p_single_seed(k, seed, instance_array, distance_calc, convergence_diff_cutoff, n_iterations))
    #put all the results in a dictionary... necessary step???
    for i in range(0,len(results)):
        multiple_run_results[seed_list[i]] = {}
        multiple_run_results[seed_list[i]]['cluster_assignments'] = results[i][1]
        multiple_run_results[seed_list[i]]['centroids'] = results[i][0]
        multiple_run_results[seed_list[i]]['q_val'] = results[i][2]
        q_val_list.append(results[i][2])
    #pick out the best answer from the runs
    print(q_val_list, np.argmin(q_val_list))
    seed_best = seed_list[np.argmin(q_val_list)]
    print('Best seed {seed}'.format(seed=seed_best))
    cluster_details = {'k_means_periodic_means':multiple_run_results[seed_best]['centroids'], 'k_means_periodic_q_val':multiple_run_results[seed_best]['q_val']}
    return multiple_run_results[seed_best]['cluster_assignments'], cluster_details

###############################################################
#############################EM-VM##############################
def _EM_VMM_check_convergence(mu_list_old, mu_list_new, kappa_list_old, kappa_list_new):
    return np.sqrt(np.sum((mu_list_old - mu_list_new)**2)), np.sqrt(np.sum((kappa_list_old - kappa_list_new)**2))

def _EM_VMM_maximise_single_cluster(input_arguments):
    cluster_ident, instance_array, assignments = input_arguments
    current_datapoints = (assignments==cluster_ident)
    print(os.getpid(), 'Maximisation step, cluster:%d, dimension:'%(cluster_ident,), end=' ')
    mu_list_cluster = []
    kappa_list_cluster = []
    n_dimensions = instance_array.shape[1]
    if np.sum(current_datapoints)>10:
        for dim_loc in range(n_dimensions):
            #print '%d'%(dim_loc),
            #kappa_tmp, loc_tmp, scale_fit = vonmises.fit(instance_array[current_datapoints, dim_loc],fscale=1)
            kappa_tmp, loc_tmp, scale_fit = EM_VMM_calc_best_fit(instance_array[current_datapoints, dim_loc])
            #update to the best fit parameters
            mu_list_cluster.append(loc_tmp)
            kappa_list_cluster.append(kappa_tmp)
        success = 1
    else:
        success = 0;mu_list_cluster = []; kappa_list_cluster = []
    print('')
    return np.array(mu_list_cluster), np.array(kappa_list_cluster),cluster_ident,success

def kappa_guess_func(kappa,R_e):
    return (R_e - spec.iv(1,kappa)/spec.iv(0,kappa))**2

def EM_VMM_calc_best_fit_optimise(z,lookup=None,N=None):
    '''Calculates MLE approximate parameters for mean and kappa for
    the von Mises distribution. Can use a lookup table for the two Bessel
    functions, or a scipy optimiser if lookup=None

    SH: 23May2013 '''
    if N is None:
        N = len(z)
    z_bar = np.sum(z)/float(N)
    mean_theta = np.angle(z_bar)
    R_squared = np.real(z_bar * z_bar.conj())
    tmp = (float(N)/(float(N)-1))*(R_squared-1./float(N))
    #This is to catch problems with the sqrt below - however, need to track down why this happens...
    #This happens for very low kappa values - i.e terrible clusters...
    if tmp<0:tmp = 0.
    R_e = np.sqrt(tmp)
    if lookup is None:
        tmp1 = opt.fmin(kappa_guess_func,3,args=(R_e,),disp=0)
        kappa = tmp1[0]
    else:
        min_arg = np.argmin(np.abs(lookup[0]-R_e))
        kappa = lookup[1][min_arg]
    return kappa, mean_theta, 1

def EM_VMM_calc_best_fit(z,N=None,lookup=None):
    '''Calculates MLE approximate parameters for mean and kappa for
    the von Mises distribution. Can use a lookup table for the two Bessel
    functions, or a scipy optimiser if lookup=None

    SH: 23May2013 '''
    if N is None:
        N = len(z)
    z_bar = np.sum(z,axis=0)/float(N)
    mean_theta = np.angle(z_bar)
    R_bar = np.abs(z_bar)
    if len(R_bar.shape)==0:
        if R_bar<0.53:
            kappa = 2.* R_bar + R_bar**3 + 5./6*R_bar**5
            #print 'approx 1'
        elif R_bar<0.85:
            kappa = -0.4 + 1.39*R_bar + 0.43/(1-R_bar)
            #print 'approx 2'
        elif R_bar<=1:
            kappa = 1./(2.*(1-R_bar))
            #print 'approx 3'
        else:
            raise ValueError()
    else:
        #kappa = R_bar*0.
        kappa = 1./(2.*(1-R_bar))
        #kappa[R_bar<=1.0] = 1./(2.*(1-R_bar[R_bar<=1.0]))
        kappa[R_bar<0.85] = -0.4 + 1.39*R_bar[R_bar<0.85] + 0.43/(1-R_bar[R_bar<0.85])
        kappa[R_bar<0.53] = 2.* R_bar[R_bar<0.53] + R_bar[R_bar<0.53]**3 + (5./6)*(R_bar[R_bar<0.53])**5
        #kappa= np.array(kappa)
    return kappa, mean_theta, 1



def EM_GMM_calc_best_fit(instance_array,weights):
    '''Calculates MLE approximate parameters for mean and kappa for
    the von Mises distribution. Can use a lookup table for the two Bessel
    functions, or a scipy optimiser if lookup=None

    SH: 23May2013 '''
    N = np.sum(weights)
    z = (instance_array.T * weights).T
    mean_theta = np.sum(z,axis=0)/float(N)
    sigma = np.sqrt(1./N *np.sum(weights[:,np.newaxis] *(instance_array - mean_theta)**2, axis = 0))
    return sigma, mean_theta



#We can do this step in parallel....
#Either parallel over clusters, or parallel over dimensions....
def _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, assignments, instance_array_complex = None, bessel_lookup_table = None, n_cpus=1):
    if instance_array_complex is None:
        instance_array_complex = np.exp(1j*instance_array)
    n_clusters = len(mu_list)
    n_datapoints, n_dimensions = instance_array.shape
    mu_list_old = copy.deepcopy(mu_list)
    kappa_list_old = copy.deepcopy(kappa_list)
    start_time = time.time()
    if n_cpus>1:
        print('creating pool map ', n_cpus)
        pool = multiprocessing.Pool(processes = n_cpus, maxtasksperchild=2)
        #output_data = pool.map(_EM_VMM_maximise_single_cluster, izip(range(n_clusters), itertools.repeat(instance_array),itertools.repeat(assignments)))
        output_data = pool.map(_EM_VMM_maximise_single_cluster, izip(range(n_clusters), itertools.repeat(instance_array),itertools.repeat(assignments)))
        pool.close(); pool.join()
        for mu_list_cluster, kappa_list_cluster, cluster_ident, success in output_data:
            if success==1:
                mu_list[cluster_ident][:]=mu_list_cluster
                kappa_list[cluster_ident][:]=kappa_list_cluster
                
    else:
        for cluster_ident in range(n_clusters):
            current_datapoints = (assignments==cluster_ident)
            #Only try to fit the von Mises distribution if there
            #are datapoints!!!!
            #print 'Maximisation step, cluster:%d, dimension:'%(cluster_ident,),
            if np.sum(current_datapoints)>0:
                for dim_loc in range(n_dimensions):
                    #print '%d-'%(dim_loc),
                    #kappa_tmp, loc_tmp, scale_fit = vonmises.fit(instance_array[current_datapoints, dim_loc],fscale=1)
                    #kappa_tmp, loc_tmp, scale_fit = calc_best_fit(instance_array[current_datapoints, dim_loc])
                    kappa_tmp, loc_tmp, scale_fit = EM_VMM_calc_best_fit(instance_array_complex[current_datapoints, dim_loc],lookup=bessel_lookup_table)
                    #update to the best fit parameters
                    mu_list[cluster_ident][dim_loc]=loc_tmp
                    kappa_list[cluster_ident][dim_loc]=kappa_tmp
    convergence_mu, convergence_kappa = _EM_VMM_check_convergence(mu_list_old, mu_list, kappa_list_old, kappa_list)
    print('maximisation time: %.2f'%(time.time()-start_time))
    return mu_list, kappa_list, convergence_mu, convergence_kappa


def _EM_VMM_maximisation_step_soft(mu_list, kappa_list, instance_array, zij, instance_array_complex = None, bessel_lookup_table = None, n_cpus=1):
    n_clusters = len(mu_list)
    n_datapoints, n_dimensions = instance_array.shape
    pi_hat = np.sum(zij,axis=0)/float(n_datapoints)
    if instance_array_complex is None:
        instance_array_complex = np.exp(1j*instance_array)
    mu_list_old = copy.deepcopy(mu_list)
    kappa_list_old = copy.deepcopy(kappa_list)

    for cluster_ident in range(n_clusters):
        inst_tmp = (instance_array_complex.T * zij[:,cluster_ident]).T
        N= np.sum(zij[:,cluster_ident])

        #calculate the best fit for this cluster - all dimensions at once.... using new approximations
        kappa_tmp1, loc_tmp1, scale_fit1 = EM_VMM_calc_best_fit(inst_tmp, lookup=bessel_lookup_table,N=N)
        mu_list[cluster_ident] = loc_tmp1
        kappa_list[cluster_ident] = kappa_tmp1
        # for dim_loc in range(n_dimensions):
        #     #Only do this if there are items to cluster!!
        #     if N>=5:
        #         kappa_tmp, loc_tmp, scale_fit = EM_VMM_calc_best_fit(inst_tmp[:, dim_loc], lookup=bessel_lookup_table,N=N)
        #         #update to the best fit parameters
        #         mu_list[cluster_ident][dim_loc]=loc_tmp
        #         kappa_list[cluster_ident][dim_loc]=kappa_tmp
        #         #print '{:.2e},{:.2e}'.format(np.max(np.abs(loc_tmp - loc_tmp1[dim_loc])), np.max(np.abs(kappa_tmp - kappa_tmp1[dim_loc]))),
        # #print ''
        # print np.max(np.abs(np.array(mu_list[cluster_ident]) - loc_tmp1)), np.max(np.abs(np.array(kappa_list[cluster_ident]) - kappa_tmp1))

    kappa_list = np.clip(kappa_list,0.1,300)
    convergence_mu, convergence_kappa = _EM_VMM_check_convergence(mu_list_old, mu_list, kappa_list_old, kappa_list)
    #print 'maximisation times: %.2f'%(time.time()-start_time)
    return mu_list, kappa_list, pi_hat, convergence_mu, convergence_kappa

def _EM_VMM_expectation_step_hard(mu_list, kappa_list, instance_array):
    start_time = time.time()
    n_clusters = len(mu_list); 
    n_datapoints, n_dimensions = instance_array.shape
    probs = np.ones((instance_array.shape[0],n_clusters),dtype=float)
    for mu_tmp, kappa_tmp, cluster_ident in zip(mu_list,kappa_list,range(n_clusters)):
        #We are checking the probability of belonging to cluster_ident
        probs_1 = np.product(np.exp(kappa_tmp*np.cos(instance_array-mu_tmp))/(2.*np.pi*spec.iv(0,kappa_tmp)),axis=1)
        probs[:,cluster_ident] = probs_1
    assignments = np.argmax(probs,axis=1)
    #return assignments, L
    return assignments, 0

def _EM_VMM_expectation_step_soft(mu_list, kappa_list, instance_array, pi_hat, c_arr = None, s_arr = None):
    n_clusters = len(mu_list); 
    n_datapoints, n_dimensions = instance_array.shape
    probs = np.ones((instance_array.shape[0],n_clusters),dtype=float)

    #c_arr and s_arr are used to speed up cos(instance_array - mu) using
    #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
    #this removes the need to constantly recalculate cos(a) and sin(a)
    if c_arr is None or s_arr is None:
        c_arr = np.cos(instance_array)
        s_arr = np.sin(instance_array)

    # c_arr2 = c_arr[:,np.newaxis,:]
    # s_arr2 = s_arr[:,np.newaxis,:]
    # pt1 = (np.cos(mu_list))[np.newaxis,:,:]
    # pt2  = (np.sin(mu_list))[np.newaxis,:,:]
    # pt2 = (c_arr2*pt1 + s_arr2*pt2)
    # pt2 = (np.array(kappa_list))[np.newaxis,:,:] * pt2
    #print pt2.shape
    for mu_tmp, kappa_tmp, p_hat, cluster_ident in zip(mu_list,kappa_list,pi_hat,range(n_clusters)):
        norm_fac_exp = len(mu_list[0])*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
        pt1 = kappa_tmp * (c_arr*np.cos(mu_tmp) + s_arr*np.sin(mu_tmp))
        probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1,axis=1) + norm_fac_exp)

        #old way without trig identity speed up
        #probs[:,cluster_ident] = p_hat * np.exp(np.sum(kappa_tmp*np.cos(instance_array - mu_tmp),axis=1)+norm_fac_exp)

        #older way including everything in exponent, and not taking log of hte constant
        #probs[:,cluster_ident] = p_hat * np.product( np.exp(kappa_tmp*np.cos(instance_array-mu_tmp))/(2.*np.pi*spec.iv(0,kappa_tmp)),axis=1)
    prob_sum = (np.sum(probs,axis=1))[:,np.newaxis]
    zij = probs/((prob_sum))

    #This was from before using np.newaxis
    #zij = (probs.T/prob_sum).T

    #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
    #unless comparing different techniques and/or checking for convergence
    valid_items = probs>1.e-20
    #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
    L = np.sum(zij[valid_items]*np.log(probs[valid_items]))
    #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))
    return zij, L


def EM_VMM_clustering(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', comment=''):
    # should really have some record of what was done here.
    #This is for the new method...
    instance_array_complex = np.exp(1j*instance_array)

    kappa_lookup = np.linspace(0,100,10000)
    bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]
    print('...')

    n_dimensions = instance_array.shape[1]
    iteration = 1    
    #First assignment step
    mu_list = np.ones((n_clusters,n_dimensions),dtype=float)
    kappa_list = np.ones((n_clusters,n_dimensions),dtype=float)
    LL_list = []
    if start=='k_means':
        print('Initialising clusters using a fast k_means run')
        cluster_assignments, cluster_details = k_means_clustering(instance_array, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 1)
        mu_list, kappa_list, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, cluster_assignments, instance_array_complex = instance_array_complex, bessel_lookup_table= bessel_lookup_table, n_cpus=n_cpus)
    elif start=='EM_GMM':
        print('Initialising clusters using a EM_GMM run')
        cluster_assignments, cluster_details = EM_GMM_clustering(instance_array, n_clusters=n_clusters, sin_cos = 0, number_of_starts = 1)
        mu_list, kappa_list, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, cluster_assignments, instance_array_complex = instance_array_complex, bessel_lookup_table = bessel_lookup_table, n_cpus=n_cpus)
    else:
        print('Initialising clusters using random start points')
        mu_list = np.random.rand(n_clusters,n_dimensions)*2.*np.pi - np.pi
        kappa_list = np.random.rand(n_clusters,n_dimensions)*20
        cluster_assignments, L = _EM_VMM_expectation_step_hard(mu_list, kappa_list,instance_array)
        while np.min([np.sum(cluster_assignments==i) for i in range(len(mu_list))])<20:#(instance_array.shape[0]/n_clusters/4):
            print('recalculating initial points')
            mu_list = np.random.rand(n_clusters,n_dimensions)*2.*np.pi - np.pi
            kappa_list = np.random.rand(n_clusters,n_dimensions)*20
            cluster_assignments = _EM_VMM_expectation_step_hard(mu_list, kappa_list,instance_array)
            print(cluster_assignments)
    convergence_record = []
    converged = 0; 
    while (iteration<=n_iterations) and converged!=1:
        start_time = time.time()
        cluster_assignments, L = _EM_VMM_expectation_step_hard(mu_list, kappa_list,instance_array)
        LL_list.append(L)
        mu_list, kappa_list, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_hard(mu_list, kappa_list, instance_array, cluster_assignments,  instance_array_complex = instance_array_complex, bessel_lookup_table = bessel_lookup_table, n_cpus=n_cpus)
        print('Time for iteration %d :%.2f, mu_convergence:%.3f, kappa_convergence:%.3f, LL: %.8e'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa,L))
        convergence_record.append([iteration, convergence_mu, convergence_kappa])
        if convergence_mu<0.01 and convergence_kappa<0.01:
            converged = 1
            print('Convergence criteria met!!')
        iteration+=1
    print('AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1]))
    cluster_details = {'EM_VMM_means':mu_list, 'EM_VMM_kappas':kappa_list, 'EM_VMM_LL':LL_list}
    return cluster_assignments, cluster_details


def EM_VMM_clustering_soft(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random',bessel_lookup_table=True):
    '''
    Expectation maximisation using von Mises with soft cluster
    assignments.  instance_array : the input phases n_clusters :
    number of clusters to aim for n_iterations : number of iterations
    before giving up n_cpus : currently not implemented start: how to
    start the clusters off - recommend using 'k_means'
    bessel_lookup_table : how to calculate kappa, can use a lookup
    table or optimiser

    SH : 23May2013
    '''

    instance_array_complex = np.exp(1j*instance_array)
    instance_array_c = np.real(instance_array_complex)
    instance_array_s = np.imag(instance_array_complex)
    n_dimensions = instance_array.shape[1]
    iteration = 1    
    if bessel_lookup_table:
        kappa_lookup = np.linspace(0,100,10000)
        bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]
    else:
        bessel_lookup_table=None
    #First assignment step
    mu_list = np.ones((n_clusters,n_dimensions),dtype=float)
    kappa_list = np.ones((n_clusters,n_dimensions),dtype=float)
    LL_list = []
    zij = np.zeros((instance_array.shape[0],n_clusters),dtype=float)
    if start=='k_means':
        print('Initialising clusters using a fast k_means run')
        cluster_assignments, cluster_details = k_means_clustering(instance_array, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 3)
        for i in list(set(cluster_assignments)):
            zij[cluster_assignments==i,i] = 1
        #print zij
        print('finished initialising')
    elif start=='EM_GMM':
        cluster_assignments, cluster_details = EM_GMM_clustering(instance_array, n_clusters=n_clusters, sin_cos = 1, number_of_starts = 1)
        for i in list(set(cluster_assignments)):
            zij[cluster_assignments==i,i] = 1
    else:
        print('going with random option')
        zij = np.random.random(zij.shape)
    mu_list, kappa_list, pi_hat, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_soft(mu_list, kappa_list, instance_array, zij, instance_array_complex = instance_array_complex, bessel_lookup_table= bessel_lookup_table, n_cpus=n_cpus)
        
    convergence_record = []
    converged = 0; 
    LL_diff = np.inf
    while (iteration<=n_iterations) and converged!=1:
        start_time = time.time()
        zij, L = _EM_VMM_expectation_step_soft(mu_list, kappa_list, instance_array, pi_hat, c_arr = instance_array_c, s_arr = instance_array_s)
        LL_list.append(L)
        cluster_assignments = np.argmax(zij,axis=1)
        mu_list, kappa_list, pi_hat, convergence_mu, convergence_kappa = _EM_VMM_maximisation_step_soft(mu_list, kappa_list, instance_array, zij, instance_array_complex = instance_array_complex, bessel_lookup_table= bessel_lookup_table, n_cpus=n_cpus)
        if (iteration>=2): LL_diff = np.abs(((LL_list[-1] - LL_list[-2])/LL_list[-2]))
        #print 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa,L,LL_diff)
        convergence_record.append([iteration, convergence_mu, convergence_kappa])
        if iteration>200 and LL_diff <0.0001:
           converged = 1
           print('Convergence criteria met!!')
        iteration+=1
    print('Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa,L,LL_diff))
    #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
    cluster_assignments = np.argmax(zij,axis=1)
    cluster_details = {'EM_VMM_means':mu_list, 'EM_VMM_kappas':kappa_list, 'EM_VMM_LL':LL_list, 'zij':zij}
    return cluster_assignments, cluster_details

def EM_VMM_clustering_wrapper2(input_data):
    tmp = EM_VMM_clustering_class(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)


def EM_VMM_clustering_wrapper(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):
    cluster_list = [n_clusters for i in range(number_of_starts)]
    seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = izip(rep(instance_array), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print('creating pool map')
        results = pool.map(EM_VMM_clustering_wrapper2, input_data_iter)
        print('waiting for pool to close ')
        pool.close()
        print('joining pool')
        pool.join()
        print('pool finished')
    else:
        results = map(EM_VMM_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: LL_results.append(tmp[1]['LL'][-1])
    print(LL_results)
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]

class EM_VMM_clustering_class():
    def __init__(self, instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None):
        '''
        Expectation maximisation using von Mises with soft cluster
        assignments.  instance_array : the input phases n_clusters :
        number of clusters to aim for n_iterations : number of iterations
        before giving up n_cpus : currently not implemented start: how to
        start the clusters off - recommend using 'k_means'
        bessel_lookup_table : how to calculate kappa, can use a lookup
        table or optimiser

        kappa_calc : approx, lookup_table, optimize
        SH : 23May2013
        '''
        #min iterations, max iterations
        #kappa change, mu change
        self.instance_array = copy.deepcopy(instance_array)
        self.instance_array_complex = np.exp(1j*self.instance_array)
        self.instance_array_c = np.real(self.instance_array_complex)
        self.instance_array_s = np.imag(self.instance_array_complex)
        self.n_instances, self.n_dimensions = self.instance_array.shape
        self.n_clusters = n_clusters
        self.max_iterations = n_iterations
        self.start = start
        self.hard_assignments = hard_assignments
        self.seed = seed
        if self.seed is None:
            self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        if kappa_calc == 'lookup_table':
            self.generate_bessel_lookup_table()
        else:
            self.bessel_lookup_table=None
        self.iteration = 1
        self.initialisation()
        self.convergence_record = []
        converged = 0; 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_VMM_expectation_step()
            if self.hard_assignments:
                print('hard assignments')
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_VMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print('Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff))
            self.convergence_record.append([self.iteration, self.convergence_mu, self.convergence_kappa])
            if (self.iteration > min_iterations) and (self.convergence_mu<mu_converged) and (self.convergence_kappa<kappa_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print('Convergence criteria met!!')
            elif self.iteration > n_iterations:
                converged = 1
                print('Max number of iterations')
            self.iteration+=1
        print(os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff))
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        #self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'LL':self.LL_list, 'zij':self.zij, 'BIC':self.BIC}
        self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'LL':self.LL_list, 'BIC':self.BIC}

    def generate_bessel_lookup_table(self):
        self.kappa_lookup = np.linspace(0,100,10000)
        self.bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]

    def initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mu_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.kappa_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.instance_array.shape[0],self.n_clusters),dtype=float)
        if self.start=='k_means':
            print('Initialising clusters using a fast k_means run')
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 3, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            print('finished initialising')
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(cluster_assignments)):
                self.zij[cluster_assignments==i,i] = 1
        else:
            print('going with random option')
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_VMM_maximisation_step()

    def _EM_VMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mu_list_old = self.mu_list.copy()
        self.kappa_list_old = self.kappa_list.copy()
        for cluster_ident in range(self.n_clusters):
            inst_tmp = (self.instance_array_complex.T * self.zij[:,cluster_ident]).T
            N= np.sum(self.zij[:,cluster_ident])
            #calculate the best fit for this cluster - all dimensions at once.... using new approximations
            self.kappa_list[cluster_ident,:], self.mu_list[cluster_ident,:], scale_fit1 = EM_VMM_calc_best_fit(inst_tmp, lookup=self.bessel_lookup_table, N=N)
        #Prevent ridiculous situations happening....
        self.kappa_list = np.clip(self.kappa_list,0.1,300)
        self._EM_VMM_check_convergence()

    def _EM_VMM_check_convergence(self,):
        self.convergence_mu = np.sqrt(np.sum((self.mu_list_old - self.mu_list)**2))
        self.convergence_kappa = np.sqrt(np.sum((self.kappa_list_old - self.kappa_list)**2))

    def _EM_VMM_expectation_step(self,):
        self.probs = self.zij*0#np.ones((self.instance_array.shape[0],self.n_clusters),dtype=float)
        #instance_array_c and instance_array_s are used to speed up cos(instance_array - mu) using
        #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        #this removes the need to constantly recalculate cos(a) and sin(a)
        for mu_tmp, kappa_tmp, p_hat, cluster_ident in zip(self.mu_list,self.kappa_list,self.pi_hat,range(self.n_clusters)):
            #norm_fac_exp = self.n_clusters*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            norm_fac_exp = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            pt1 = kappa_tmp * (self.instance_array_c*np.cos(mu_tmp) + self.instance_array_s*np.sin(mu_tmp))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1,axis=1) + norm_fac_exp)
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)
        #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
        #unless comparing different techniques and/or checking for convergence
        #This is to prevent problems with log of a very small number....
        #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
        #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))







def EM_GMM_clustering_wrapper2(input_data):
    tmp = EM_GMM_clustering_class(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)


def EM_GMM_clustering_wrapper(instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):
    cluster_list = [n_clusters for i in range(number_of_starts)]
    seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = izip(rep(instance_array), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print('creating pool map')
        results = pool.map(EM_GMM_clustering_wrapper2, input_data_iter)
        print('waiting for pool to close ')
        pool.close()
        print('joining pool')
        pool.join()
        print('pool finished')
    else:
        results = map(EM_GMM_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: LL_results.append(tmp[1]['LL'][-1])
    print(LL_results)
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]

class EM_GMM_clustering_class():
    def __init__(self, instance_array, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx',hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None):
        '''
        Expectation maximisation using von Mises with soft cluster
        assignments.  instance_array : the input phases n_clusters :
        number of clusters to aim for n_iterations : number of iterations
        before giving up n_cpus : currently not implemented start: how to
        start the clusters off - recommend using 'k_means'
        bessel_lookup_table : how to calculate kappa, can use a lookup
        table or optimiser

        kappa_calc : approx, lookup_table, optimize
        SH : 23May2013
        '''
        #min iterations, max iterations
        #kappa change, mu change
        self.instance_array = copy.deepcopy(instance_array)
        #self.instance_array_complex = np.exp(1j*self.instance_array)
        #self.instance_array_c = np.real(self.instance_array_complex)
        #self.instance_array_s = np.imag(self.instance_array_complex)
        self.n_instances, self.n_dimensions = self.instance_array.shape
        self.n_clusters = n_clusters
        self.max_iterations = n_iterations
        self.start = start
        self.hard_assignments = hard_assignments
        self.seed = seed
        if self.seed is None:
            self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        self.iteration = 1
        self.initialisation()
        self.convergence_record = []
        converged = 0; 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_GMM_expectation_step()
            if self.hard_assignments:
                print('hard assignments')
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_GMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print('Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff))
            self.convergence_record.append([self.iteration, self.convergence_mu, self.convergence_kappa])
            if (self.iteration > min_iterations) and (self.convergence_mu<mu_converged) and (self.convergence_kappa<kappa_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print('Convergence criteria met!!')
            elif self.iteration > n_iterations:
                converged = 1
                print('Max number of iterations')
            self.iteration+=1
        print(os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff))
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.std_list, 'LL':self.LL_list, 'zij':self.zij, 'BIC':self.BIC}

    def initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mu_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.std_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.instance_array.shape[0],self.n_clusters),dtype=float)
        if self.start=='k_means':
            print('Initialising clusters using a fast k_means run')
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 3, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            print('finished initialising')
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(cluster_assignments)):
                self.zij[cluster_assignments==i,i] = 1
        else:
            print('going with random option')
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_GMM_maximisation_step()

    def _EM_GMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mu_list_old = self.mu_list.copy()
        self.std_list_old = self.std_list.copy()
        for cluster_ident in range(self.n_clusters):
            #inst_tmp = (self.instance_array.T * self.zij[:,cluster_ident]).T
            #N= np.sum(self.zij[:,cluster_ident])
            #calculate the best fit for this cluster - all dimensions at once.... using new approximations
            self.std_list[cluster_ident,:], self.mu_list[cluster_ident,:] = EM_GMM_calc_best_fit(self.instance_array, self.zij[:,cluster_ident])
        #Prevent ridiculous situations happening....
        self.std_list = np.clip(self.std_list,0.001,300)
        self._EM_VMM_check_convergence()

    def _EM_VMM_check_convergence(self,):
        self.convergence_mu = np.sqrt(np.sum((self.mu_list_old - self.mu_list)**2))
        self.convergence_kappa = np.sqrt(np.sum((self.std_list_old - self.std_list)**2))

    def _EM_GMM_expectation_step(self,):
        self.probs = self.zij*0#np.ones((self.instance_array.shape[0],self.n_clusters),dtype=float)
        #instance_array_c and instance_array_s are used to speed up cos(instance_array - mu) using
        #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        #this removes the need to constantly recalculate cos(a) and sin(a)
        for mu_tmp, std_tmp, p_hat, cluster_ident in zip(self.mu_list,self.std_list,self.pi_hat,range(self.n_clusters)):
            #norm_fac_exp = self.n_clusters*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            #norm_fac_exp = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(spec.iv(0,std_tmp)))
            norm_fac_exp = self.n_dimensions*np.log(1./np.sqrt(2.*np.pi)) + np.sum(np.log(1./std_tmp))
            #pt1 = kappa_tmp * (self.instance_array_c*np.cos(mu_tmp) + self.instance_array_s*np.sin(mu_tmp))
            pt1 = -(self.instance_array - mu_tmp)**2/(2*(std_tmp**2))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1,axis=1) + norm_fac_exp)
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)
        #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
        #unless comparing different techniques and/or checking for convergence
        #This is to prevent problems with log of a very small number....
        #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
        #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))



#############################################################################
#####################Plotting functions#####################################
def plot_alfven_lines_func(ax):
    pickled_theoretical = pickle.load(file('theor_value.pickle','r'))
    mu = 4.*np.pi*(10**(-7))
    mi = 1.673*(10**(-27))
    meff = 2.5
    lamda = 0.27
    colors = ['k--','b-','r','y','g']
    for plotting_mode in range(0,2):
        B = 0.5
        va  = B/np.sqrt((10.**18) * meff * mi * mu)*lamda
        #ax.plot(pickled_theoretical['kh_list'],np.array(pickled_theoretical['k_par'][plotting_mode])*va/(2.*np.pi),'c-',linewidth=5)
        ax.plot(pickled_theoretical['kh_list'],np.array(pickled_theoretical['k_par'][plotting_mode])*va/(2.*np.pi),colors[plotting_mode],linewidth=4)
        #B = 0.5*5/7
        #va  = B/np.sqrt((10.**18) * meff * mi * mu)*lamda
        #ax.plot(pickled_theoretical['kh_list'],np.array(pickled_theoretical['k_par'][plotting_mode])*va/(2.*np.pi),'--')

def test_von_mises_fits():
    N = 20000
    mu = 2.9
    kappa = 5
    mu_list = np.linspace(-np.pi,np.pi,20)
    kappa_list = range(1,50)
    mu_best_fit = []
    mu_record = []
    fig,ax = pt.subplots(nrows = 2)
    def kappa_guess_func(kappa,R_e):
        return (R_e - spec.iv(1,kappa)/spec.iv(0,kappa))**2

    def calc_best_fit(theta):
        z = np.exp(1j*theta)
        z_bar = np.average(z)
        mean_theta = np.angle(z_bar)
        R_squared = np.real(z_bar * z_bar.conj())
        R_e = np.sqrt((float(N)/(float(N)-1))*(R_squared-1./float(N)))
        tmp1 = opt.fmin(kappa_guess_func,3,args=(R_e,))
        return mean_theta, tmp1[0]

    for mu in mu_list:
        kappa_best_fit = []
        for kappa in kappa_list:
            mu_record.append(mu)
            theta = np.random.vonmises(mu,kappa,N)
            mean_theta, kappa_best = calc_best_fit(theta)
            mu_best_fit.append(mean_theta)
            kappa_best_fit.append(kappa_best)
        ax[0].plot(kappa_list, kappa_best_fit,'.')
        ax[0].plot(kappa_list, kappa_list,'-')
    ax[1].plot(mu_record,mu_best_fit,'.')
    ax[1].plot(mu_record,mu_record,'-')
    fig.canvas.draw(); fig.show()

##INS def generate_artificial_covar_data(n_clusters

def generate_artificial_data(n_clusters, n_dimensions, n_instances, prob=None, method='vonMises', means=None, variances=None, random_means_bounds = [-np.pi,np.pi], random_var_bounds = [0.05,5]):
    '''Generate a dummy data set n_clusters : number of separate
    clusters n_dimensions : how many seperate phase signals per
    instance n_instances : number of instances - note this might be
    changed slightly depending on the probabilities

    kwargs prob : 1D array, length n_clusters, probabilty of each
    cluster - (note must add to one...)  if None, then all clusters
    will have equal probability method : distribution to draw points
    from - vonMises or Gaussian means, variances: arrays (n_clusters x
    n_dimensions) of the means and variances (kappa for vonMises) for
    the distributions if these are given, n_clusters and n_dimensions
    are ignored If only means or variances are given, then the missing
    one will be given random values Note for vonMises, 1/var is used
    as this is approximately kappa

    SH : 14May2013 '''
    if means!=None and variances!=None:
        if means.shape!=variances.shape:
            raise ValueError('means and variances have different shapes')
        n_clusters, n_dimensions = means.shape
    if means!=None: n_clusters, n_dimensions = means.shape
    if variances!=None: n_clusters, n_dimensions = means.shape

    #randomly generate the means and variances using uniform distribution and the bounds given
    if means is None:
        means = np.random.rand(n_clusters,n_dimensions)*(random_means_bounds[1]-random_means_bounds[0]) + random_means_bounds[0]
    if variances is None:
        variances = np.random.rand(n_clusters,n_dimensions)*(random_var_bounds[1]-random_var_bounds[0]) + random_var_bounds[0]
    print(means)
    print(variances)
    #figure out how many instances per cluster based on the probabilities
    if prob is None:
        prob = np.ones((n_clusters,),dtype=float)*1./(n_clusters)
    elif np.abs(np.sum(prob)-1)>0.001:
        raise ValueError('cluster probabilities dont add up to one within 0.001 tolerance....')
    n_instances_per_clust = np.array(map(int, (np.array(prob)*n_instances)))
    input_data = np.zeros((n_instances, n_dimensions),dtype=float)
    cluster_assignments = np.zeros((n_instances,),dtype=int)

    start_point = 0; end_point = 0
    for i in range(n_clusters):
        end_point = end_point + n_instances_per_clust[i]
        for j in range(n_dimensions):
            #input_data[start_point:end_point,j] = vonmises.rvs(variances[i,j],size=n_instances_per_clust[i],loc=means[i,j],scale=1)
            if method=='vonMises':
                input_data[start_point:end_point,j] = vonmises.rvs(variances[i,j],size=n_instances_per_clust[i],loc=means[i,j],scale=1)
            elif method=='Gaussian':
                input_data[start_point:end_point,j] = norm.rvs(size=n_instances_per_clust[i],loc=means[i,j],scale=1./variances[i,j]*3)
            else:
                raise ValueError('method must be either vonMises or Gaussian')
        cluster_assignments[start_point:end_point] = i
        start_point = end_point

    #Move data into [-pi,pi]
    input_data = input_data %(2.*np.pi)    
    input_data[input_data>np.pi] -= 2.*np.pi

    #shuffle the rows of the data so they
    #aren't in order of the clusters which might cause some problems....
    locs = np.arange(n_instances)
    np.random.shuffle(locs)
    input_data = input_data[locs,:]
    cluster_assignments = cluster_assignments[locs]
    feat_obj = feature_object(instance_array=input_data, misc_data_dict={})#, misc_data_labels):
    
    #create a clustering object with all info, and put it as the first object in the
    #clustered items list
    tmp = clustering_object()
    tmp.settings = {'method':'EM_VMM'}
    tmp.cluster_assignments = cluster_assignments
    tmp.cluster_details = {'EM_VMM_means':means,'EM_VMM_kappas':variances}
    tmp.feature_obj = feat_obj
    feat_obj.clustered_objects.append(tmp)
    return feat_obj

def make_grid_subplots(n_subplots, sharex = 'all', sharey = 'all'):
    '''This helper function generates the many subplots
    on a regular grid

    SH: 23May2013
    '''
    n_cols = int(math.ceil(n_subplots**0.5))
    if n_subplots/float(n_cols)>n_subplots//n_cols:
        n_rows = n_subplots//n_cols + 1
    else:
        n_rows = n_subplots//n_cols
    fig, ax = pt.subplots(nrows = n_rows, ncols = n_cols, sharex = 'all', sharey = 'all'); ax = ax.flatten()
    #fig, ax = pt.subplots(nrows = n_rows, ncols = n_cols,subplot_kw=dict(projection='polar')); ax = ax.flatten()
    return fig, ax

def modtwopi(x, offset=np.pi):
    """ return an angle in the range of offset +-
    2pi>>> print("{0:.3f}".format(modtwopi( 7),offset=3.14))0.717
    This simple strategy works when the number is near zero +- 2Npi,
    which is true for calculating the deviation from the cluster centre.
    does not attempt to make jumps small (use fix2pi_skips for that)
    """
    return ((-offset+np.pi+np.array(x)) % (2*np.pi) +offset -np.pi)

def convert_kappa_std(kappa,deg=True):
    '''This converts kappa from the von Mises distribution into a
    standard deviation that can be used to generate a similar normal
    distribution

    SH: 14June2013
    '''
    R_bar = spec.iv(1,kappa)/spec.iv(0,kappa)
    if deg==True:
        return np.sqrt(-2*np.log(R_bar))*180./np.pi
    else:
        return np.sqrt(-2*np.log(R_bar))

def EM_VMM_GMM_clustering_wrapper2(input_data):
    tmp = EM_VMM_GMM_clustering_class(*input_data)
    return copy.deepcopy(tmp.cluster_assignments), copy.deepcopy(tmp.cluster_details)

def EM_VMM_GMM_clustering_wrapper(instance_array, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, number_of_starts = 1):

    cluster_list = [n_clusters for i in range(number_of_starts)]
    seed_list = [i for i in range(number_of_starts)]
    rep = itertools.repeat
    from multiprocessing import Pool
    input_data_iter = izip(rep(instance_array), rep(instance_array_amps), rep(n_clusters),
                                     rep(n_iterations), rep(n_cpus), rep(start), rep(kappa_calc),
                                     rep(hard_assignments), rep(kappa_converged),
                                     rep(mu_converged),rep(min_iterations), rep(LL_converged),
                                     rep(verbose), seed_list)
    if n_cpus>1:
        pool_size = n_cpus
        pool = Pool(processes=pool_size, maxtasksperchild=3)
        print('creating pool map')
        results = pool.map(EM_VMM_GMM_clustering_wrapper2, input_data_iter)
        print('waiting for pool to close ')
        pool.close()
        print('joining pool')
        pool.join()
        print('pool finished')
    else:
        results = map(EM_VMM_GMM_clustering_wrapper2, input_data_iter)
    LL_results = []
    for tmp in results: LL_results.append(tmp[1]['LL'][-1])
    print(LL_results)
    tmp_loc = np.argmax(LL_results)
    return results[tmp_loc]


class EM_VMM_GMM_clustering_class(clustering_object):
    def __init__(self, instance_array, instance_array_amps, n_clusters = 9, n_iterations = 20, n_cpus=1, start='random', kappa_calc='approx', hard_assignments = 0, kappa_converged = 0.1, mu_converged = 0.01, min_iterations=10, LL_converged = 1.e-4, verbose = 0, seed=None):
        '''This model is supposed to include a mixture of Gaussian and von
        Mises distributions to allow datamining of data that essentially
        consists of complex numbers (amplitude and phase) such as most
        Fourier based measurements. Supposed to be an improvement on the
        case of just using the phases between channels - more interested
        in complex modes such as HAE, and also looking at data that is
        more amplitude based such as line of sight chord through the
        plasma for interferometers and imaging diagnostics.

        Note the amplitude data is included in
        misc_data_dict['mirnov_data'] from the stft-clustering
        extraction technique

        Need to figure out a way to normalise it... so that shapes of
        different amplitudes will look the same
        Need to plumb this in somehow...

        SH: 15June2013
        '''
        self.settings = {'n_clusters':n_clusters,'n_iterations':n_iterations,'n_cpus':n_cpus,'start':start,
                         'kappa_calc':kappa_calc,'hard_assignments':hard_assignments, 'method':'EM_VMM_GMM'}
        self.instance_array = copy.deepcopy(instance_array)
        self.instance_array_amps = np.abs(instance_array_amps)
        norm_factor = np.sum(self.instance_array_amps,axis=1)
        self.instance_array_amps = self.instance_array_amps/norm_factor[:,np.newaxis]

        self.instance_array_complex = np.exp(1j*self.instance_array)
        self.instance_array_c = np.real(self.instance_array_complex)
        self.instance_array_s = np.imag(self.instance_array_complex)
        self.n_instances, self.n_dimensions = self.instance_array.shape
        self.n_dimensions_amps = self.instance_array_amps.shape[1]
        self.n_clusters = n_clusters
        self.max_iterations = n_iterations
        self.start = start
        self.hard_assignments = hard_assignments
        self.seed = seed
        if self.seed is None:
            self.seed = os.getpid()
        print('seed,',self.seed)
        np.random.seed(self.seed)
        if kappa_calc == 'lookup_table':
            self.generate_bessel_lookup_table()
        else:
            self.bessel_lookup_table=None
        self.iteration = 1
        self.initialisation()
        self.convergence_record = []
        converged = 0 
        self.LL_diff = np.inf
        while converged!=1:
            start_time = time.time()
            self._EM_VMM_GMM_expectation_step()
            if self.hard_assignments:
                print('hard assignments')
                self.cluster_assignments = np.argmax(self.zij,axis=1)
                self.zij = self.zij *0
                for i in range(self.n_clusters):
                    self.zij[self.cluster_assignments==i,i] = 1

            valid_items = self.probs>(1.e-300)
            self.LL_list.append(np.sum(self.zij[valid_items]*np.log(self.probs[valid_items])))
            self._EM_VMM_GMM_maximisation_step()
            if (self.iteration>=2): self.LL_diff = np.abs(((self.LL_list[-1] - self.LL_list[-2])/self.LL_list[-2]))
            if verbose:
                print('Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff))
            self.convergence_record.append([self.iteration, self.convergence_mu, self.convergence_kappa])
            if (self.iteration > min_iterations) and (self.convergence_mu<mu_converged) and (self.convergence_kappa<kappa_converged) and (self.LL_diff<LL_converged):
                converged = 1
                print('Convergence criteria met!!')
            elif self.iteration > n_iterations:
                converged = 1
                print('Max number of iterations')
            self.iteration+=1
        print(os.getpid(), 'Time for iteration %d :%.2f, mu_convergence:%.3e, kappa_convergence:%.3e, LL: %.8e, LL_dif : %.3e'%(self.iteration,time.time() - start_time,self.convergence_mu, self.convergence_kappa,self.LL_list[-1],self.LL_diff))
        #print 'AIC : %.2f'%(2*(mu_list.shape[0]*mu_list.shape[1])-2.*LL_list[-1])
        self.cluster_assignments = np.argmax(self.zij,axis=1)
        self.BIC = -2*self.LL_list[-1]+self.n_clusters*3*np.log(self.n_dimensions)
        #self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'LL':self.LL_list, 'zij':self.zij, 'BIC':self.BIC}
        self.cluster_details = {'EM_VMM_means':self.mu_list, 'EM_VMM_kappas':self.kappa_list, 'EM_GMM_means':self.mean_list, 'EM_GMM_std':self.std_list, 'LL':self.LL_list, 'BIC':self.BIC}

    def generate_bessel_lookup_table(self):
        self.kappa_lookup = np.linspace(0,100,10000)
        self.bessel_lookup_table = [spec.iv(1,kappa_lookup)/spec.iv(0,kappa_lookup), kappa_lookup]

    def initialisation(self):
        '''This involves generating the mu and kappa arrays
        Then initialising based on self.start using k-means, EM-GMM or
        giving every instance a random probability of belonging to each cluster
        SH: 7June2013
        '''
        self.mu_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.mean_list = np.ones((self.n_clusters,self.n_dimensions_amps),dtype=float)
        self.kappa_list = np.ones((self.n_clusters,self.n_dimensions),dtype=float)
        self.std_list = np.ones((self.n_clusters,self.n_dimensions_amps),dtype=float)
        self.LL_list = []
        self.zij = np.zeros((self.instance_array.shape[0],self.n_clusters),dtype=float)
        #maybe only the random option is valid here.....
        if self.start=='k_means':
            print('Initialising clusters using a fast k_means run')
            self.cluster_assignments, self.cluster_details = k_means_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 3, seed=self.seed)
            for i in list(set(self.cluster_assignments)):
                self.zij[self.cluster_assignments==i,i] = 1
            print('finished initialising')
        elif self.start=='EM_GMM':
            self.cluster_assignments, self.cluster_details = EM_GMM_clustering(self.instance_array, n_clusters=self.n_clusters, sin_cos = 1, number_of_starts = 1)
            for i in list(set(cluster_assignments)):
                self.zij[cluster_assignments==i,i] = 1
        else:
            print('going with random option')
            #need to get this to work better.....
            self.zij = np.random.random(self.zij.shape)
            #and normalise so each row adds up to 1....
            self.zij = self.zij / ((np.sum(self.zij,axis=1))[:,np.newaxis])
        self._EM_VMM_GMM_maximisation_step()

    def _EM_VMM_GMM_maximisation_step(self):
        '''Maximisation step SH : 7June2013
        '''
        self.pi_hat = np.sum(self.zij,axis=0)/float(self.n_instances)
        self.mu_list_old = self.mu_list.copy()
        self.kappa_list_old = self.kappa_list.copy()
        self.mean_list_old = self.mean_list.copy()
        self.std_list_old = self.std_list.copy()

        for cluster_ident in range(self.n_clusters):
            inst_tmp = (self.instance_array_complex.T * self.zij[:,cluster_ident]).T
            N= np.sum(self.zij[:,cluster_ident])
            #calculate the best fit for this cluster - all dimensions at once.... using new approximations
            #VMM part
            self.kappa_list[cluster_ident,:], self.mu_list[cluster_ident,:], scale_fit1 = EM_VMM_calc_best_fit(inst_tmp, lookup=self.bessel_lookup_table, N=N)
            #GMM part
            self.std_list[cluster_ident,:], self.mean_list[cluster_ident,:] = EM_GMM_calc_best_fit(self.instance_array_amps, self.zij[:,cluster_ident])
        #Prevent ridiculous situations happening....
        self.kappa_list = np.clip(self.kappa_list,0.1,300)
        self.std_list = np.clip(self.std_list,0.001,300)
        self._EM_VMM_GMM_check_convergence()

    def _EM_VMM_GMM_check_convergence(self,):
        self.convergence_mu = np.sqrt(np.sum((self.mu_list_old - self.mu_list)**2))
        self.convergence_kappa = np.sqrt(np.sum((self.kappa_list_old - self.kappa_list)**2))
        self.convergence_mean = np.sqrt(np.sum((self.mean_list_old - self.mean_list)**2))
        self.convergence_std = np.sqrt(np.sum((self.std_list_old - self.std_list)**2))

    def _EM_VMM_GMM_expectation_step(self,):
        self.probs = self.zij*0
        #instance_array_c and instance_array_s are used to speed up cos(instance_array - mu) using
        #the trig identity cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        #this removes the need to constantly recalculate cos(a) and sin(a)
        for mu_tmp, kappa_tmp, mean_tmp, std_tmp, p_hat, cluster_ident in zip(self.mu_list,self.kappa_list,self.mean_list, self.std_list, self.pi_hat,range(self.n_clusters)):
            #norm_fac_exp = self.n_clusters*np.log(1./(2.*np.pi)) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            norm_fac_exp_VMM = -self.n_dimensions*np.log(2.*np.pi) - np.sum(np.log(spec.iv(0,kappa_tmp)))
            norm_fac_exp_GMM = self.n_dimensions_amps*np.log(1./np.sqrt(2.*np.pi)) + np.sum(np.log(1./std_tmp))
            pt1_GMM = -(self.instance_array_amps - mean_tmp)**2/(2*(std_tmp**2))
            pt1_VMM = kappa_tmp * (self.instance_array_c*np.cos(mu_tmp) + self.instance_array_s*np.sin(mu_tmp))
            self.probs[:,cluster_ident] = p_hat * np.exp(np.sum(pt1_VMM,axis=1) + norm_fac_exp_VMM +
                                                         np.sum(pt1_GMM,axis=1) + norm_fac_exp_GMM)
        prob_sum = (np.sum(self.probs,axis=1))[:,np.newaxis]
        self.zij = self.probs/(prob_sum)
        #Calculate the log-likelihood - note this is quite an expensive computation and not really necessary
        #unless comparing different techniques and/or checking for convergence
        #This is to prevent problems with log of a very small number....
        #L = np.sum(zij[probs>1.e-20]*np.log(probs[probs>1.e-20]))
        #L = np.sum(zij*np.log(np.clip(probs,1.e-10,1)))

