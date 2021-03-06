"""
A quick test - this version is newer than cluster_DA_0.py
_PYFUSION_TEST_@@DAfilename='$DA/300_384_RMSv2_neNBecVA.npz' cl=-1
"""
from pyfusion.data.DA_datamining import DA
import pylab as pl

DAfilename = '/data/datamining/PF2_130813_50_5X_1.5_5b_rms_1_diags.npz'
DAfilename='../../../datamining/dd/300_384_RMSv2_neNBecVA.npz'
#DAfilename='DAnov1516diag.npz'
#DAfilename='/data/datamining/PF2_130813_6X_1.5_5b_rms_1_diags.npz'
# too big - 8GB for phase - DAfilename='/data/datamining/PF2_130813_8X_1.5_5b_rms_1._diags.npz'
import pyfusion.clustering as clust

_var_defaults="""
phase_sign = 1        # NEVER leave this at -1 - too dangerous
number_of_starts = 2
max_instances=50000
n_clusters=6
n_iterations=10       # 10 is not really enough - but a good start for a demo     
n_cpus=2
keysel=None
sel=None
cl=0
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if phase_sign != 1: print('**** Warning! - you are fiddling with the phase ****')

(phases, misc) = clust.convert_DA_file(DAfilename,keysel=keysel,limit=max_instances)
phases = phase_sign*phases  # -1 to compare with boyd's code sep 2013 on H-1
if sel is not None:
    print('Selecting before clustering ')
    phases = phases[sel,:]  # WAS phases[:,sel] - how did it ever work?
    for k in misc.keys(): misc.update({k: misc[k][sel]}) # this was missing too

fo = clust.feature_object(phases, misc)
co = fo.cluster(method='EM_VMM',n_clusters = n_clusters, n_iterations = n_iterations,start = 'k_means',n_cpus=n_cpus,number_of_starts = number_of_starts)

co.plot_clusters_phase_lines()  # show clusters
# extract the meta data corresponding to the instances selected
DA(DAfilename,load=1).extract(locals(),inds=co.feature_obj.misc_data_dict['serial'])

while cl >= 0:
    cl = int(input('\n  cluster number to view, -1 or ^C to quit? '))
    w = co.members(cl)
    pl.figure('cluster {cl}'.format(cl=cl))
    pl.scatter(t_mid[w],freq[w],500*amp[w])
    pl.ylim(0,max(freq)/2)
    pl.show(0)
"""
# simplest possible version

run pyfusion/examples/medium_300.py
import pyfusion.clustering as clust
phases=DA300.da.pop('phases')
fo = clust.feature_object(phases, DA300.da)
co = fo.cluster(method='EM_VMM',n_clusters = 6,n_iterations = 10,start = 'k_means',n_cpus=1,number_of_starts = 2)
"""

