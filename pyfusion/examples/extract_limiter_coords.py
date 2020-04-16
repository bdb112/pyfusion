"""
From the component library of W7X, extract the coordinates of the
node list of the limiter, count the unique points, and extract those
lying in the plane of the limiter given by Phi=4/5 * pi

run pyfusion/examples/extract_limiter_coords.py "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest/component/483/"
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json
import pyfusion
from pyfusion.utils.time_utils import utc_GMT,utc_ns
 
# http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest/component/482/data


# I used to use 474, but 485 seems more recent (both for m5)
"""
http://webservices.ipp-hgw.mpg.de/docs/componentsDbRest.html#searchMeshModels
485: Inboard limiter for OP1.1 in m 5. Created 12.06.2014, used for EPS 2014. Limiter designed at phi=0, for configuration (1, 1, 1, 1, 1, 0.13, 0.13) at +9 cm from copper cooling structure. The shaping is ideal limiter for P = 10 MW/m2, assuming decay length of 15 mm and parallel flux of 40 MW/m2. Grid size theta/phi: 81/91. Stepping is along local surface normals. The 0-surface is based on Fourier coefficients max poloidal number M=30, max toroidal number N=21, calculated with Biot-Savart step = 1cm. Additional modifications: parabolic and plateau inserts in the center, linear extension at the edges. The limiter poloidal extent theta=(-.485, .485).
474: Inboard limiter in module 5. Limiter designed for OP1.1 at phi = 0, for configuration (1, 1, 1, 1, 1, 0.2, 0.2) at +5.3 cm from copper cooling structure. The shaping is ideal limiter for P = 10 MW/m2m assuming decay length of 3.55 mm and parallel flux of 200 MW/m2. Stepping is along local surface normals. The 0-surface is based on Fourier coefficients max poloidal number M=40, max toroidal number N=40, calculated with Biot-Savart step=2mm. The limiter poloidal extent theta=(-.411, .411) toprevent a small third zone in connection length. Local naming: +5.3cm_v8.
"""
component_path = sys.argv[1] if len(sys.argv) > 1 else '/data/databases/W7X/LP/limiter_component_483'
if 'http' in component_path:
    if sys.version < '3.0.0':
        from future.standard_library import install_aliases
        install_aliases()

    from urllib.request import urlopen, Request
    nfname = component_path
    comp = json.load(urlopen(os.path.join(component_path, 'data'), timeout=pyfusion.TIMEOUT))
    info = json.load(urlopen(os.path.join(component_path, 'info'), timeout=pyfusion.TIMEOUT))
    print(info['comment'])
    print([utc_GMT(entry['timeStamp'] *1000000000) for entry in info['history']])
    if 'surfaceMesh' in comp:
        dat = comp['surfaceMesh']
        nodes = dat['nodes']
        x, y, z = [np.array(nodes[_]) for _ in ['x1','x2','x3']]
        i = dat['nodeIds']
        tris = dat['polygons']
else:
    info = {}
    nfname = component_path + '.nodes.bz2'
    i, x, y, z = np.loadtxt(nfname).T
    efname = component_path + '.elements.bz2'
    tris = np.loadtxt(efname, dtype=int)

print('{np:,} vertices of which {nu:,} are unique, {nq:,} polygons in object'
      .format(np=len(x), nq=len(tris),
              nu=len(np.unique([hash(nd_) for nd_ in zip(x, y, z)]))))

th = 2*np.pi*4/5.  # rotate until limiter is at phi=0
X, Y, Z = np.cos(th)*x + np.sin(th)*y, np.sin(th)*x - np.cos(th)*y, z
# filter out the phi=0 points - luckily there are some very close in most, but not all
wY0 = np.where(np.abs(Y) < np.sort(np.abs(Y))[100])[0]  # get the 100 smallest abs Y
wZ0 = np.where(np.abs(Z) < np.sort(np.abs(Z))[100])[0]  # Z=0 pts ar 5mm either side
#  tried to visualize the Z=0.2 plane but we need +/- 0.02 becuse of tilt of grid?
plt.plot(X[wY0], Z[wY0], '+')
plt.title('vertical profile of centre line')
  # remove the 4x duplication due to quadrilaterals - fails if  there are more in wY0 then wZ0
zu, inds = np.unique(z[wY0], return_index=1)
xu = X[wY0][inds]
plt.plot(xu, zu, '+')

topview = plt.figure()
plt.plot(X[wZ0], Y[wZ0], ',') # use a dot as we need to include many points

plt.figure()
axdR = plt.gca()
# .0001 is needed to get the Y=0 point otherwise derivative flips sign at the origin
wq = np.where((Y[wZ0] > -0.0001) & (X[wZ0]-5.6686 > -0.0108))[0]
axdR.plot(Y[wZ0[wq]][::5], X[wZ0[wq]][::5]-5.6686, '.', label='limiter_profile')
axdR.set_aspect('equal')
axdR.set_ylabel('R relative to limiter tip')
xx = np.linspace(0, 0.06, 300)
pfit = np.polyfit(Y[wZ0[wq]], X[wZ0[wq]]-5.6686, 12)
# 12th order is wiggly, but the best so far.  Could 'straighten' the bit 
# near the origin first?
axdR.plot(xx, np.polyval(pfit, xx),label='poly fit to profile')
axdR.set_title('midplane section')
axdR.set_xlabel('transverse distance')
axdR.legend(loc='upper left')

axsin=axdR.twinx()
pfitder = np.polyder(pfit)
axsin.plot(xx, np.sin(np.arctan(np.polyval(pfitder, xx))),
           label=r'$sin(\alpha)$ from an order {nth} poly'.format(nth=len(pfit)-1))
axsin.legend(loc='upper right')
axsin.plot([0, axsin.get_xlim()[1]], [0, 0], 'k', lw=0.5)
axsin.set_ylim(axsin.get_ylim()[0],0.2)
# dR means delta Rmaj
plt.show(0)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# for tr in tris[0:1000]-1:           # this works for the first few hundred,
#     plt.plot(x[tr],y[tr],z[tr],'k') # then we get spurious lines
plt.plot(x, y, z, '.', label=nfname)

plt.legend(prop={'size': 'small'})
#  use this to force aspect equal
from pyfusion.visual.threed_aspect import set_axes_equal
set_axes_equal(ax)

# from six.moves import input
# ans = input('Write a json coording file to temp dir? y/N')
# if 'y' in ans.lower():
import json
try:
    thispath = os.path.dirname(__file__)
    W7X_path = os.path.realpath(thispath+'/../acquisition/W7X/')
except NameError as reason:
    print("Using current dir as __file__ not understood", reason.__repr__())
    W7X_path = '.'
    
fh = file(os.path.join(W7X_path,'limiter_geometry.json'), 'wt')
small_data = dict(rlim=xu.tolist(), zlim=zu.tolist(), shape_poly=pfit.tolist(), fname=nfname, info=info)
# a sample of the points along the centreline - presumably the wetted line
dnsample = 2
small_data.update(dict(xvcl=x[wY0][inds][::dnsample].tolist(),
                       yvcl=y[wY0][inds][::dnsample].tolist(),
                       zvcl=z[wY0][inds][::dnsample].tolist(),
                       xhcl=x[wZ0][inds][::dnsample].tolist(),
                       yhcl=y[wZ0][inds][::dnsample].tolist(),
                       zhcl=z[wZ0][inds][::dnsample].tolist(),
))
json.dump(small_data, fh)
print('saved in ' + fh.name)

"""
# paint the component surface according to flux coordinate
from pyfusion.acquisition.W7X.SOL_distance import distance_to_surface, vmec, Client, Points3D
uxyz, uinds = unique([hash(nd) for nd in zip(x,y,z)], return_index=1)
# redo uxyz to perhaps make the order smarter
uxyz = np.array([x, y, z]).T[uinds]
v = Points3D(uxyz.T)
vmc = vmec.service.toVMECCoordinates('w7x_ref_113', Points3D([norm([v.x1,v.x2],axis=0), arctan2(v.x2, v.x1), v.x3]), tolerance=1e-6)
import matplotlib.tri as tri
deci=2
triang = tri.Triangulation(Y[uinds][::deci], Z[uinds][::deci])
# need to fold in the sin(alpha)
plt.tripcolor(triang, -clip(array(vmc.x1[::deci]),.65,.68),shading='gouraud')
"""
