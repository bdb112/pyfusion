"""
From the component library of W7X, extract the coordinates of the
node list of the limiter, count the unique points, and extract those
lying in the plane of the limiter given by Phi=4/5 * pi
"""
import numpy as np
import matplotlib.pyplot as plt
fname = '/data/databases/W7X/LP/limiter_component_474.nodes.bz2'
i, x, y, z = np.loadtxt(fname).T
print('{np:,} points in object, {nu:,} are unique'
      .format(np=len(x), nu=len(np.unique(zip(x, y, z)))))

th = 2*np.pi*4/5.  # rotate until limiter is at phi=0
X, Y = np.cos(th)*x + np.sin(th)*y, np.sin(th)*x - np.cos(th)*y
w0 = np.where(np.abs(Y) < .00001)[0]  # filter out the phi=0 points - luckily there are some very close
plt.plot(X[w0], z[w0], '+')
zu, inds = np.unique(z[w0], return_index=1)  # remove the 4x duplication
xu = X[w0][inds]
plt.plot(xu, zu, '+')
import json
json.dump(dict(rlim=xu.tolist(), zlim=zu.tolist()), file('limiter_crescent.json', 'wt'))


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(x,y,z,'.',label=fname)

plt.legend(prop={'size':'small'})
## use this to force aspect equal
from threed import set_axes_equal
set_axes_equal(ax)

