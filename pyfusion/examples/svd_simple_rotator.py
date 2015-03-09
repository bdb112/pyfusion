""" simple SVD example/test case independent of pyfusion 
to aid with interpetation of pyfusion SVDs 
""" 

from numpy import arange, sin, cos, pi, transpose, array, diag, allclose, dot
from numpy.linalg import svd
import pylab as pl
import numpy as np
_var_default = """
t=arange(0,1e-4,1e-6)
w=2*pi*20e3
hold=0
xrLHDMP = np.deg2rad([18, 90, 126, 198, 270, 342])
xrLHDHMP= np.deg2rad([-75.39, -54.39, -43.68, -21.96, -10.99, 0.0, 10.99, 21.96, 43.68, 54.39, 75.39, 85.64, 95.73])
xr = None
shift=0  # shift=1 is a 1/npol shift
am=0
harm=[0]
m=1  #  can be fractional, but what does this mean?
npol=30
svmax = 100  # limit the number of singular vectors used in reconstruction.
"""

# tweak above parameters according to command line args
exec(_var_default)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())


dat2d = []  # prepare to accummulate channels
if xr is None:
    xr = arange(0,2*pi,2*pi/npol)
    xr[0]=shift*xr[1] + (1-shift)*xr[0]
for x in xr: 
    varr=(1+am*cos(w*t/5))*(cos(w*t-m*x) + #harm[0]*cos(w*2*t-m*2*x))
                            np.sum(np.array([harm[i]*cos(w*(i+2)*t-m*x*(i+2)) for i in range(len(harm))]),0))
    dat2d.append(varr)
if hold==0: pl.clf()

pl.subplot(2,2,1)
pl.plot(transpose(dat2d) ,hold=1)
pl.title('original data')
[topo,svs,chrono] = svd(dat2d) # do the SVD with full matrices

pl.subplot(2,2,2)  # plot the top reconstructed
nused = min(len(svs),svmax)
for c in (0, npol/2):
    pl.plot(sum([array(topo[:,i])*svs[i]*chrono[i,c] for i in range(nused)],0), hold=1)
    pl.plot(transpose(dat2d)[c],'--',linewidth=2)
pl.title('data[0] vs position reconstructed from {n} SVS'.format(n=nused))

SVS=',  ' .join(["{s:.3g}".format(s=s) for s in svs[0:6]])
print(SVS)
pl.suptitle('SVS = '+SVS)
[topof,svsf,chronof] = svd(dat2d,full_matrices=False)
print allclose(dat2d, dot(topof,dot(diag(svsf),chronof)))
print(' ' .join(["{s:8.2g}".format(s=s) for s in svsf[0:8]]))
pl.subplot(2,2,3)
pl.plot(transpose(chronof[0:3,:]),hold=0)
pl.plot(transpose(chronof[3:6,:]),linewidth=0.5,hold=1)
pl.title('top 3 chronos')

pl.subplot(2,2,4)
pl.plot(topo[:,0:3],hold=0)
pl.plot(topo[:,3:6],linewidth=0.5,hold=1)
pl.title('top 3 topos')
pl.show()
