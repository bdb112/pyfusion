# http://webservices.ipp-hgw.mpg.de/docs/vmec.html#getReff
from osa import Client
from numpy.linalg import norm
from matplotlib import pyplot as plt

def Points3D(arr):  #convenience function - NOT the same as vmec.types.Points3D 
    p = vmec.types.Points3D()
    p.x1, p.x2, p.x3 = arr
    return(p)

vmec = Client('http://esb:8280/services/vmec_v8?wsdl')
# note: vmec_v5 is in all the examples.  v8 wont work with toCylinderCoords
# v4,6,7, also exist.  7 works with toCylinderCoords

p = vmec.types.Points3D()
p.x1 = [6.0]
p.x2 = [0.0]
p.x3 = [0.0]

reff = vmec.service.getReff('w7x_ref_81', p)
print(reff)

import numpy as np
vmec_coords = vmec.types.Points3D()

npts = 1000 #  69ms for 1,000
vmec_coords.x1 = np.linspace(0.4, 0.4, num=npts)
vmec_coords.x2 = np.linspace(-0.1, 0.1, num=npts)
vmec_coords.x3 = np.linspace(-0.1, 0.1, num=npts)

cyl = vmec.service.toCylinderCoordinates('w7x_ref_113', vmec_coords)

# limiter vertical centreline from extract_limiter coordinates.
import json
# this file is written by 
small_data = json.load(file('pyfusion/devices/W7X/limiter_geometry.json','r'))
limcl = np.array([small_data['xvcl'], small_data['yvcl'], small_data['zvcl']]).T
v = Points3D(limcl.T)
limVMEC = vmec.service.toVMECCoordinates('w7x_ref_113', Points3D([norm([v.x1,v.x2],axis=0), np.arctan2(v.x2, v.x1), v.x3]), tolerance=1e-6)
plt.plot(limcl.T[2],limVMEC.x1)
plt.show(block=0)
