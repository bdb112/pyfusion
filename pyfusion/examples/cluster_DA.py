"""
run pyfusion/examples/medium_300.py
DA_file=DA300.name
"""
from pyfusion.data.DA_datamining import DA, report_mem
DA_file = '/data/datamining/PF2_130813_50_5X_1.5_5b_rms_1_diags.npz'
DA_file='../../../datamining/dd/300_384_RMSv2_neNBecVA.npz'
import pyfusion.clustering as clust
(phases, misc) = clust.convert_DA_file(DA_file,limit=50000)
fo = clust.feature_object(phases, misc)
# 10 iterations is not enough (50 is better), but this is just a demo.
co = fo.cluster(method='EM_VMM',n_clusters = 6,n_iterations = 10,start = 'k_means',n_cpus=2,number_of_starts = 2)


"""
run pyfusion/examples/medium_300.py
import pyfusion.clustering as clust
phases=DA300.da.pop('phases')
fo = clust.feature_object(phases, DA300.da)
co = fo.cluster(method='EM_VMM',n_clusters = 6,n_iterations = 10,start = 'k_means',n_cpus=1,number_of_starts = 2)
"""
