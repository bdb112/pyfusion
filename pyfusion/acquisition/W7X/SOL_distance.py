from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from numpy import pi
from osa import Client

vmec = Client('http://esb:8280/services/vmec_v5?wsdl')
# Some reference equilibria
"""
Timing:
P_LP11 w7x_ref_113, sval=0.6534784
[20,20], maxits=8 -> 500ms to 2.7e-9
[30,30], maxits=7 -> 547ms to 2.7e-9
[40,40], maxits=6 -> 656ms to 2.7e-9
[50,50], maxits=6 -> 891ms to 2.7e-9
# one gui advantage of 50x50 is that you can see the whole torus

# limiter centrelins at equaotr corresponds for sval=0.654871
surf_distance, dic = distance_to_surface(sval=0.654871, vmec_eq='w7x_ref_113',point=limcl[8], npts=[30,30],ax3D=None,max_its=6)
0.636 is 3mm away from lim cl
All points are > 3.11 mm so if we use 0.654871, all distances are positive.
# So now plot the distance from last surface to the limiter centreline
# this is not the whole story for flat faced limiters such as 483 (OK for 474)

SOL = [[pt[2],distance_to_surface(sval=0.654871,vmec_eq='w7x_ref_113',point=pt, ax3D=None,max_its=4)[0]] for pt in limcl]
print '\n'.join([str('{0:8.2f}, {1:8.2f}'.format(*(pair*1e3))) for pair in array(SOL)])

EEM - OP1.1. limiter; config. J w7x/1000_1000_1000_1000_+0390_+0390/01/020s/ w7x_ref_60 
EEM - OP1.1. limiter; config. J w7x/1000_1000_1000_1000_+0390_+0390/01/00s/ w7x_ref_59  
EEM - OP1.1. limiter; config. J w7x/1000_1000_1000_1000_+0390_+0390/01/00l4/ w7x_ref_113 
GGP - OP1.1. limiter; config. J index 13 w7x/1000_1000_1000_1000_+0390_+0000/05/0433/ is ref_81
# vmec_eq = 'w7x_ref_81'  # used in python examples
"""

def Points3D(arr):
    p = vmec.types.Points3D()
    p.x1, p.x2, p.x3 = arr
    return(p)

def Cyl2XYZ(R, Phi=None, Z=None):
    """ should work for Point3D and 3 vecs """
    if Phi is None and hasattr(R, 'x1'):
        R, Phi, Z = R.x1, R.x2, R.x3
    X, Y = R * np.cos(Phi), R * np.sin(Phi)
    return(np.array([X, Y, Z]))

def dist2surf_iter(sval=0.6535, point=[1.7565, -5.4059, -0.2214], vmec_eq='w7x_ref_113', npts=[30,30], theta_range=[-pi,pi], phi_range=[-pi, pi], ax3D=None):
    """ returns closest 50 indices, XYZ and vmc arrays
    """
    vmc = vmec.types.Points3D()
    
    t0, t1 = theta_range
    p0, p1 = phi_range
    s, theta, phi = np.mgrid[sval:sval:1j,t0:t1:npts[0]*1j, p0:p1:npts[1]*1j]
    dims = np.shape(s[0])
    s, theta, phi = s[0].flatten(), theta[0].flatten(), phi[0].flatten()
    vmc.x1, vmc.x2, vmc.x3 = s, theta, phi
    # get the corresponding points in RPhiZ
    cyl = vmec.service.toCylinderCoordinates(vmec_eq, vmc)
    RPhiZ =  np.array([cyl.x1, cyl.x2, cyl.x3])
    R, Phi, Z = RPhiZ # convenience vars
    XYZ = Cyl2XYZ(R, Phi, Z)

    if ax3D is not None:
        ax3D.plot(*XYZ, linestyle='None', marker=',')
        ax3D.plot(*(np.array([point]).T), marker='*')
                                                                            
    closest = np.argsort(LA.norm(point - XYZ.T, axis=1))
    surf_distance = LA.norm(point - XYZ[:, closest[0]])
    print('s=', sval, ', closesest is', surf_distance, XYZ[:, closest[0]])
    print('SOL depth for', dims[0], 'x', dims[1], 'grids = ',
          np.min(LA.norm(point - XYZ.T, axis=1)))
    return(surf_distance, dict(closest=closest, XYZ=XYZ, vmc=vmc))

           
def distance_to_surface(sval=0.6534784, point=[1.7565, -5.4059, -0.2214], vmec_eq='w7x_ref_113', npts=[30,30], npts_init=None, theta_range=[-pi,pi], phi_range=[-pi, pi],max_its=4, ax3D=None):
    """ iterates dist2surf_iter to return distance, and dict with closest 50 indices, XYZ and vmc arrays
    If ax3D is None, done plot;  1: make 3D axes; otherwise use existing
    """
    if ax3D is not None and ax3D == 1:
        ax3D = plt.figure().add_subplot(111, projection='3d')
    it = max_its - 1
    if npts_init is None:
        npts_init = [30,200]  # probably could adjust this according to aspect/ntor
    surf_distance, dic = dist2surf_iter(sval=sval, vmec_eq=vmec_eq, npts=npts_init, point = point, ax3D=None if it>2 else ax3D)

    for it in range(max_its)[::-1]:
        vmc = dic['vmc']
        closest = dic['closest']
        s, theta, phi = vmc.x1, vmc.x2, vmc.x3
        thet0 = theta[closest[0]]
        phi0 = phi[closest[0]]
        dtheta = np.diff(np.unique(theta))[0]
        dphi = np.diff(np.unique(phi))[0]

        surf_distance, dic = dist2surf_iter(theta_range=[thet0-dtheta, thet0+dtheta],
                                            phi_range=[phi0-dphi, phi0+dphi],
                                            sval=sval, vmec_eq=vmec_eq, npts=npts,
                                            point=point, ax3D=None if it>2 else ax3D)

    if ax3D is not None:
        from pyfusion.visual.threed_aspect import set_axes_equal
        set_axes_equal(ax3D)
        plt.show()
        
    retdic = dic
    return(surf_distance, retdic)


if __name__ == "__main__":
    
    hold = False
    if not hold:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax3D = fig.add_subplot(111, projection='3d')
    else:
        ax3D = None

    P_LP11 = np.array([1.75650, -5.40590, -0.22140])


    surf_distance, dic = distance_to_surface(sval=0.6535, vmec_eq='w7x_ref_113',
                                             point=P_LP11, ax3D=ax3D)
"""
Usage
ax3D = figure().add_subplot(111, projection='3d')

# Check limiter corresponds to a surface 
# run pyfusion/examples/extract_limiter_coords.py c:/cygwin/tmp/component_474 # old
run pyfusion/examples/extract_limiter_coords.py "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest/component/483/"
figure()
sval=0.6534784; vmec_eq='w7x_ref_113'
limcl=array([small_data['xvcl'], small_data['yvcl'], small_data['zvcl']]).T
SOL = array([[limcl[2], distance_to_surface(sval=sval, vmec_eq=vmec_eq, point=pt, ax3D=None, max_its=4)[0]] for pt in limcl])
XYZ = Cyl2XYZ(vmec.service.toCylinderCoordinates(vmec_eq, Points3D([sval, 0, 0])))
reff = vmec.service.getReff(vmec_eq, Points3D(XYZ))[0]
plot(limcl[:,2], SOL[:,1], label=str('distance to reff={reff:.4f}m'.format(reff=float(reff))))
xlabel('Z(m)'); suptitle(nfname); legend(loc='best')


# Lukas' distances after running W7X_OP1.1_LCFS.py
run pyfusion/examples/W7X_OP1.1_LCFS.py
sval=0.6534784; vmec_eq='w7x_ref_113'
#sval=0.6743; vmec_eq='w7x_ref_114'
SOL = array([[LP[1], distance_to_surface(sval=sval, vmec_eq=vmec_eq,point=LP[2], ax3D=None, max_its=4)[0]] for LP in allLP])
figure()
plot [LP[1] for LP in allLP],  [LP[-1] for LP in allLP],'*r',label='Lukas',markersize=20
plot SOL[:,0],SOL[:,1]*1000,'o',label='VMEC:'+vmec_eq + str(',s={0:.4f}'.format(sval)), ms=8
legend loc='best'
title('SOL distance in mm for both LP sets')
show()
xyzMPM13=array([-2.1486, -5.66423, -0.168])

"""
