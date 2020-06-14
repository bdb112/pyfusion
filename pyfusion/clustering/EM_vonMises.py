'''
bdb's version of SH implementation of EM von Mises (or something like it...)
1/ use_DA sets access to data in DA rather than pickles
2/ multiprocessing (n_cpus>1) is impossible to stop cleanly 

See pyfusion/examples/clusterDA.py for another attempt

SH: 2May2013
'''

#from multiprocessing import Queue, Process, Pool
import multiprocessing
from scipy.optimize import fmin_slsqp
import copy, pickle, time, itertools, os
import scipy.cluster.hierarchy as hclust
from scipy.stats.distributions import vonmises
import numpy as np
import matplotlib.pyplot as pt
#not sure what the shape parameter is
#plot a few different PDFs

###########################
#Important settings
########################
input_phase_differences = '5MHz_7MHz_combined_data.pickle'#'misc_data_test.pickle'
input_misc_data = '5MHz_7MHz_combined_misc.pickle'#'misc_data_test.pickle'
n_iters = 20 #iterations before stopping
#n_iters = 5 #iterations before stopping
n_cpus = 3 #Number of CPUs to use for calculation
n_clusters = 3


subset=False
use_DA = True

#load data....
if use_DA:
    from pyfusion.data.DA_datamining import DA
    da=DA('DAMIRNOV_41_13_BEST_LOOP_10_3ms_20180912043.npz')
    da.info()
    expt_data = da['phases']
    misc_data = da
    misc_output_labels = da.keys() 
    for key in 'phases,info'.split(','):
        misc_output_labels.remove(key)
    misc_output_data = np.array([da[key] for key in misc_output_labels])
else:
    expt_data = np.array(pickle.load(file(input_phase_differences,'r')))
    misc_output_data = pickle.load(file(input_misc_data,'r'))
    misc_output_labels = misc_output_data['labels']
    misc_output_data = np.array(misc_output_data['data'])

if subset:
    misc_output_data = misc_output_data[::50]
    expt_data = expt_data[::50]
    print("Using a subset")

#kh_plot_item = misc_output_labels.index('kh')
freq_plot_item = misc_output_labels.index('freq')
expt_data[expt_data>np.pi]-=2.*np.pi
data_points, n_dimensions = expt_data.shape
em_input_X = expt_data*1.
EM_output_filename = input_phase_differences.rstrip('.pickle')+'output_EM_vonMises_%d.pickle'%(n_clusters)

def expectation_step(mu_list, kappa_list, em_input_X):
    start_time = time.time()
    n_clusters = len(mu_list); 
    n_datapoints, n_dimensions = em_input_X.shape
    probs = np.ones((em_input_X.shape[0],n_clusters),dtype=float)
    for mu_tmp, kappa_tmp, cluster_ident in zip(mu_list,kappa_list,range(n_clusters)):
        #We are checking the probability of belonging to cluster_ident
        for mu, kappa, dim_loc in zip(mu_tmp, kappa_tmp,range(n_datapoints)):
            #we are looking at each dimension individually and multiplying them out
            fit_dist = vonmises(kappa,loc=mu)
            probs[:,cluster_ident] = probs[:,cluster_ident]*fit_dist.pdf(em_input_X[:,dim_loc])
    assignments = np.argmax(probs,axis=1)
    print 'Expectation step time : %.2f'%(time.time() - start_time), iteration, [np.sum(assignments==i) for i in range(len(mu_list))]
    return assignments

def check_convergence(mu_list_old, mu_list_new, kappa_list_old, kappa_list_new):
    return np.sqrt(np.sum((mu_list_old - mu_list_new)**2)), np.sqrt(np.sum((kappa_list_old - kappa_list_new)**2))


def maximise_single_cluster(input_arguments):
    cluster_ident, em_input_x, assignments = input_arguments
    current_datapoints = (assignments==cluster_ident)
    print os.getpid(), 'Maximisation step, cluster:%d, dimension:'%(cluster_ident,),
    mu_list_cluster = []
    kappa_list_cluster = []
    n_dimensions = em_input_X.shape[1]
    if np.sum(current_datapoints)>10:
        for dim_loc in range(n_dimensions):
            print '%d'%(dim_loc),
            kappa_tmp, loc_tmp, scale_fit = vonmises.fit(em_input_X[current_datapoints, dim_loc],fscale=1)
            #update to the best fit parameters
            mu_list_cluster.append(loc_tmp)
            kappa_list_cluster.append(kappa_tmp)
        success = 1
    else:
        success = 0;mu_list_cluster = []; kappa_list_cluster = []
    print ''
    return np.array(mu_list_cluster), np.array(kappa_list_cluster),cluster_ident,success

#We can do this step in parallel....
#Either parallel over clusters, or parallel over dimensions....
def maximisation_step(mu_list, kappa_list, em_input_X, assignments, cpus=1):
    n_clusters = len(mu_list)
    n_datapoints, n_dimensions = em_input_X.shape
    mu_list_old = copy.deepcopy(mu_list)
    kappa_list_old = copy.deepcopy(kappa_list)
    start_time = time.time()
    if cpus>1:
        print 'creating pool map ', cpus
        pool = multiprocessing.Pool(processes = cpus, maxtasksperchild=2)
        output_data = pool.map(maximise_single_cluster, itertools.izip(range(n_clusters), itertools.repeat(em_input_X),itertools.repeat(assignments)))
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
            print 'Maximisation step, cluster:%d, dimension:'%(cluster_ident,),
            if np.sum(current_datapoints)>0:
                for dim_loc in range(n_dimensions):
                    print '%d'%(dim_loc),
                    kappa_tmp, loc_tmp, scale_fit = vonmises.fit(em_input_X[current_datapoints, dim_loc],fscale=1)
                    #update to the best fit parameters
                    mu_list[cluster_ident][dim_loc]=loc_tmp
                    kappa_list[cluster_ident][dim_loc]=kappa_tmp
            print ''
    convergence_mu, convergence_kappa = check_convergence(mu_list_old, mu_list, kappa_list_old, kappa_list)
    print 'maximisation time: %.2f'%(time.time()-start_time)
    del em_input_X
    return mu_list, kappa_list, convergence_mu, convergence_kappa

#try fitting some data using AH

###  main #######

#initial guesses for cluster centres
mu_list = np.random.rand(n_clusters,n_dimensions)*2.*np.pi - np.pi
kappa_list = np.random.rand(n_clusters,n_dimensions)*20
iteration = 1    
#First assignment step
assignments = expectation_step(mu_list, kappa_list,em_input_X)

while np.min([np.sum(assignments==i) for i in range(len(mu_list))])<20:#(em_input_X.shape[0]/n_clusters/4):
    print 'recalculating initial points'
    mu_list = np.random.rand(n_clusters,n_dimensions)*2.*np.pi - np.pi
    kappa_list = np.random.rand(n_clusters,n_dimensions)*20
    assignments = expectation_step(mu_list, kappa_list,em_input_X)
    print assignments
convergence_record = []
converged = 0; 
while (iteration<=n_iters) and converged!=1:
    start_time = time.time()
    assignments = expectation_step(mu_list, kappa_list,em_input_X)
    mu_list, kappa_list, convergence_mu, convergence_kappa = maximisation_step(mu_list, kappa_list, em_input_X, assignments, cpus=n_cpus)
    print 'Time for iteration %d :%.2f, mu_convergence:%.3f, kappa_convergence:%.3f'%(iteration,time.time() - start_time,convergence_mu, convergence_kappa)
    convergence_record.append([iteration, convergence_mu, convergence_kappa])
    if convergence_mu<0.01 and convergence_kappa<0.01:
        converged = 1
        print 'Convergence criteria met!!'
    iteration+=1

save_clusters = 0
tmp = {'assignments':assignments, 'mu':mu_list,'kappa':kappa_list,'data':em_input_X,
       'misc_labels':misc_output_labels, 'misc_data':misc_output_data}
if save_clusters:
    pickle.dump(tmp, file(EM_output_filename,'w'))

