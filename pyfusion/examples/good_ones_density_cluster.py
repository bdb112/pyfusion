from pyfusion.data.DA_datamining import DA
da=DA('/mythextra/datamining/DA300_384_rms1_b5a_f32_diags.npz')
time run  -i pyfusion/examples/cluster_phases_three_colour.py DAfilename='/mythextra/datamining/DA300_384_rms1_b5a_f32_diags.npz'
cl
cl=26
time run  -i pyfusion/examples/cluster_phases_three_colour.py DAfilename='/mythextra/datamining/DA300_384_rms1_b5a_f32_diags.npz'
d_big=.5
run -i pyfusion/examples/density_cluster.py n_bins=70 n_plot=600 scale=.03 method='unsafe' dphase=1 clmax=35 sel=range(11,16)
clgrey=-1
run -i pyfusion/examples/density_cluster.py n_bins=70 n_plot=600 scale=.03 method='unsafe' dphase=1 clmax=35 sel=range(11,16) clgrey=-1
savez_compressed('DA300_32_50639_600_70_bins',clinds=clinds, subset=subset, subset_counts=subset_counts,dphase=dphase,sel=sel,n_bins=n_bins,cmd="run -i pyfusion/examples/density_cluster.py n_bins=70 n_plot=600 scale=.03 method='unsafe' dphase=1 clmax=35 sel=range(11,16) clgrey=-1")
clusterfile
clusterfile='DA300_32_50639_600_70_bins'
time run  -i pyfusion/examples/cluster_phases_three_colour.py DAfilename='/mythextra/datamining/DA300_384_rms1_b5a_f32_diags.npz' cl=0 d_big=1.2 clusterfile='DA300_32_50639_600_70_bins'

import pyximport; pyximport.install()
import pyfusion.utils
import pyfusion.utils.dist_mp
import pyfusion.utils.dist_nogil


x=np.load(clusterfile)
for k in x.keys(): exec("{v}=x['{k}']".format(v=k,k=k))

subset=subset[:,sel]
frlow=0
frhigh=1e10

run -i  pyfusion/examples/cluster_phases_three_colour.py DAfilename='/2TBRAID/datamining/54_117_MP2012_384_rms_1_f32_part_params_114.npz' clusterfile='pyfusion/ideal_toroidal_modes.npz' cl=9

run -i x=np.load(clusterfile)
sel=arange(5)
subset=subset[:,sel]

pyfusion/examples/cluster_phases_three_colour.py DAfilename='/2TBRAID/datamining/all54_116_diags613bsign.npz' clusterfile='pyfusion/ideal_toroidal_modes.npz' cl=8 d_big=0.8




