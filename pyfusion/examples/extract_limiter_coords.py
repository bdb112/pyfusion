"""
From the component library of W7X, extract the coordinates of the
node list of the limiter, count the unique points, and extract those
lying in the plane of the limiter given by Phi=4/5 * pi
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys

component_path = sys.argv[1] if len(sys.argv) > 1 else '/data/databases/W7X/LP/limiter_component_474.'

nfname = component_path + 'nodes.bz2'
i, x, y, z = np.loadtxt(nfname).T
efname = component_path + 'elements.bz2'
tris = np.loadtxt(efname, dtype=int)
print('{np:,} vertices, {nq:,} quadrilaterals in object, {nu:,} are unique'
      .format(np=len(x), nq=len(tris), nu=len(np.unique(zip(x, y, z)))))

th = 2*np.pi*4/5.  # rotate until limiter is at phi=0
X, Y, Z = np.cos(th)*x + np.sin(th)*y, np.sin(th)*x - np.cos(th)*y, z
# filter out the phi=0 points - luckily there are some very close
wY0 = np.where(np.abs(Y) < .00001)[0]  
wZ0 = np.where(np.abs(Z) < .005)[0]  # Z=0 pts ar 5mm either side
#  tried to visualize the Z=0.2 plane but we need +/- 0.02 becuse of tilt of grid?
plt.plot(X[wY0], Z[wY0], '+')
plt.title('vertical profile of centre line')
zu, inds = np.unique(z[wY0], return_index=1)  # remove the 4x duplication due to quadrilaterals
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
from threed_aspect import set_axes_equal
set_axes_equal(ax)

# from six.moves import input
# ans = input('Write a json coording file to temp dir? y/N')
# if 'y' in ans.lower():
import json
thispath = os.path.dirname(__file__)
W7X_path = os.path.realpath(thispath+'/../acquisition/W7X/')
fh = file(os.path.join(W7X_path,'limiter_geometry.json'), 'wt')
small_data = dict(rlim=xu.tolist(), zlim=zu.tolist(), shape_poly=pfit.tolist())
# a sample of the points along the centreline - presumably the wetted line
small_data.update(dict(xcl=x[inds][::10].tolist(),ycl=y[inds][::10].tolist(), zcl=z[inds][::10].tolist()))
json.dump(small_data, fh)
xcl=x[inds][::10]
print('saved in ' + fh.name)
