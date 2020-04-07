from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from numpy import pi
from osa import Client

vmec = Client('http://esb:8280/services/vmec_v5?wsdl')
# Some reference equilibria
"""
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

def distance_to_surface(sval=0.6535, point=[1.7565, -5.4059, -0.2214], vmec_eq='w7x_ref_113', npts=[30,30], theta_range=[-pi,pi], phi_range=[-pi, pi], ax3D=None):
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
    X, Y = R * np.cos(Phi), R * np.sin(Phi)
    XYZ = np.array([X, Y, Z])
    if ax3D is not None:
        ax3D.plot(X,Y,Z,',')
        ax3D.plot(*(np.array([point]).T), marker='*')
                                                                            
    closest = np.argsort(LA.norm(P_LP11 - XYZ.T, axis=1))
    surf_distance = LA.norm(point - XYZ[:, closest[0]])
    print('s=', sval, ', closeset is', surf_distance, XYZ[:, closest[0]])
    print('SOL depth for', dims[0], 'x', dims[1], 'grids = ',
          np.min(LA.norm(P_LP11 - XYZ.T, axis=1)))
    return(surf_distance, dict(closest=closest, XYZ=XYZ, vmc=vmc))

           
hold = False
if not hold:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
else:
    ax3D = None

P_LP11 = np.array([1.75650, -5.40590, -0.22140])

surf_distance, dic = distance_to_surface(npts=[30,200], point = P_LP11, ax3D=None)

for it in range(6)[::-1]:
    vmc = dic['vmc']
    closest = dic['closest']
    s, theta, phi = vmc.x1, vmc.x2, vmc.x3
    thet0 = theta[closest[0]]
    phi0 = phi[closest[0]]
    dtheta = np.diff(np.unique(theta))[0]
    dphi = np.diff(np.unique(phi))[0]

    surf_distance, dic = distance_to_surface(theta_range=[thet0-dtheta, thet0+dtheta],
                                             phi_range=[phi0-dphi, phi0+dphi],
                                             point=P_LP11, ax3D=None if it>2 else ax3D)

from pyfusion.visual.threed_aspect import set_axes_equal
set_axes_equal(ax3D)
plt.show()
