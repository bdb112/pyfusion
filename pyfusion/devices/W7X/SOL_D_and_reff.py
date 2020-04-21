import pyfusion
import sys
import numpy as np

# sys.path.insert(0, 'c:\\PortablePrograms\\winpythonlibs')
from pyfusion.data.read_zip_data import read_MPM_data
from pyfusion.acquisition.W7X.SOL_distance import distance_to_surface, vmec, Client, Points3D

# calculate the SOL distance for the given MPM plunge
upto = None  # use 3 to speed up for tests
if not 'allLP' in locals():
    raise('run -i is needed and run pyfusion/examples/W7X_OP1.1_LCFS.py')

LPxyz = array([lp[-2] for lp in allLP[:upto]]).T

dat = read_MPM_data( 'W7X/MPM/MPM_20160309_13.zip'); sval=0.6534784; vmec_eq='w7x_ref_113'
#dat = read_MPM_data( 'W7X/MPM/MPM_20160309_32.zip'); sval=0.6743; vmec_eq='w7x_ref_114'
vars = dict(vmec_eq=vmec_eq, sval=sval, mshot=dat['shot'][0]%1e6*1000 + dat['shot'][1])

xyz = np.array([dat['x'],dat['y'],dat['z'][0:540]])

SOL = np.array([[pt, distance_to_surface(sval=sval, vmec_eq=vmec_eq, point=pt, ax3D=None, max_its=4)[0]] for pt in xyz.T[:upto]])

fname = str('SOL_{vmec_eq}_{sval:.6f}_{mshot:.0f}.txt'.format(**vars))
with open(fname,'w') as fh:
    fh.writelines([str('{0:.4f}\n'.format(xx[1])) for xx in SOL])

# Get the reffs for plunge
reff_plunge = vmec.service.getReff(vmec_eq, Points3D(xyz))
reff_fname = str('reff_{vmec_eq}_{mshot:.0f}.txt'.format(**vars))

with open(reff_fname,'w') as fh:
    fh.writelines([str('{0:.4f}\n'.format(xx)) for xx in reff_plunge])

# get the reffs for probes
# slowly, but can keep relation to the probe
# reff_LP = [[lp[0], lp[1], vmec.service.getReff(vmec_eq, Points3D(lp[-2]))[0]] for lp in allLP]
# save reff_LP[-1]

# efficiently
reff_LP = vmec.service.getReff(vmec_eq, Points3D(LPxyz))

reff_fname = str('reff_LP_{vmec_eq}_{mshot:.0f}.txt'.format(**vars))

with open(reff_fname,'w') as fh:
    fh.writelines([str('{0:.4f}\n'.format(xx)) for xx in reff_LP])

# get the SOL distances for probes

SOL_LP = np.array([[pt, distance_to_surface(sval=sval, vmec_eq=vmec_eq, point=pt, ax3D=None, max_its=4)[0]] for pt in LPxyz.T[:upto]])

SOL_LP_fname = str('SOL_LP_{vmec_eq}_{mshot:.0f}.txt'.format(**vars))

with open(SOL_LP_fname,'w') as fh:
    fh.writelines([str('{0:.4f}\n'.format(xx[-1])) for xx in SOL_LP])

"""
with open(SOL_LP_fname.replace('SOL','XYZ'),'w') as fh:
    fh.writelines([str('{0:.4f} {0:.4f} {0:.4f}\n'.format(*xx)) for xx in LPxyz.T])
"""
