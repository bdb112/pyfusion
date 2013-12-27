"""
run pyfusion/examples/medium_300.py
DAfilename=DA300.name
"""
from pyfusion.data.DA_datamining import DA, report_mem
DAfilename = '/data/datamining/PF2_130813_50_5X_1.5_5b_rms_1_diags.npz'
DAfilename='../../../datamining/dd/300_384_RMSv2_neNBecVA.npz'
#DAfilename='DAnov1516diag.npz'

#DAfilename='/data/datamining/PF2_130813_6X_1.5_5b_rms_1_diags.npz'
# too big - 8GB for phase - DAfilename='/data/datamining/PF2_130813_8X_1.5_5b_rms_1._diags.npz'
import pyfusion.clustering as clust

_var_default="""
phase_sign = 1              # NEVER leave this at -1 - too dangerous
number_of_starts = 2
max_instances=50000
n_clusters=6
n_iterations=10
n_cpus=2
keysel=None
sel=None
"""
exec(_var_default)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if phase_sign != 1: print('**** Warning! - you are fiddling with the phase ****')

(phases, misc) = clust.convert_DA_file(DAfilename,keysel=keysel,limit=max_instances)
phases = phase_sign*phases  # -1 to compare boyd's code sep 2013 on H-1
if sel is not None:
    phases=phases[:,sel]

fo = clust.feature_object(phases, misc)
# 10 iterations is not enough (50 is better), but this is just a demo.
co = fo.cluster(method='EM_VMM',n_clusters = n_clusters, n_iterations = n_iterations,start = 'k_means',n_cpus=n_cpus,number_of_starts = number_of_starts)


"""
run pyfusion/examples/medium_300.py
import pyfusion.clustering as clust
phases=DA300.da.pop('phases')
fo = clust.feature_object(phases, DA300.da)
co = fo.cluster(method='EM_VMM',n_clusters = 6,n_iterations = 10,start = 'k_means',n_cpus=1,number_of_starts = 2)
"""
