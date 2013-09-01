run pyfusion/examples/medium_300.py
import pyfusion.clustering as clust
(phases, misc) = clust.convert_DA_file(DA300.name,limit=10000)
fo = clust.feature_object(phases, misc)
co = fo.cluster(method='EM_VMM',n_clusters = 6,n_iterations = 10,start = 'k_means',n_cpus=1,number_of_starts = 2)


"""
run pyfusion/examples/medium_300.py
import pyfusion.clustering as clust
phases=DA300.da.pop('phases')
fo = clust.feature_object(phases, DA300.da)
co = fo.cluster(method='EM_VMM',n_clusters = 6,n_iterations = 10,start = 'k_means',n_cpus=1,number_of_starts = 2)
"""
