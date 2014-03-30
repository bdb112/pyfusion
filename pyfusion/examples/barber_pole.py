""" Draws a barber pole for a particular n, m and compares with the LHD arrays
So far only works in real space - should really be in magnetic coordinates.

"""

import pylab as pl
import numpy as np
pi = np.pi

_var_defaults="""

res = 50 # number of points along each dimension
N = 3
M = 1
ms=6
hold=0
"""
exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

def eff(th, phH):
    return(np.sin(2*pi-th-phH)/(np.cos(th)*np.sin(phH)))

phi, theta = np.mgrid[-.5: .5: res*1j, 0: 1: 2*res*1j]
wave = np.cos(2*pi*M*theta - 2*pi*N*phi)
pl.imshow(wave, extent = [np.min(phi), np.max(phi), np.min(theta), np.max(theta)],hold=hold)
#if cm is None: 
#    cm = pl.jet
#cm()
pl.title('N={N}, M={M}'.format(N=N, M=M))
pl.show()

MPPhi = np.array([ 0.31415927,  1.57079633,  2.19911486,  3.45575192,  4.71238898, 5.96902604])
MPTheta = np.array([ 1.70571028,  1.70571028,  1.70571028,  1.70571028,  1.70571028, 1.70571028])

HMPPhi = np.array([ 5.27089434,  5.20108117,  5.16617459,  5.09636142,  5.06145483,\
        5.02654825,  4.99164166,  4.95673508,  4.88692191,  4.85201532,\
        4.78220215,  4.74729557,  4.71238898])

HMPTheta = np.array([-1.31580372, -0.94928458, -0.76235982, -0.3832743 , -0.19181168,\
        0.        ,  0.19181168,  0.3832743 ,  0.76235982,  0.94928458,\
        1.31580372,  1.49469997,  1.67080369])

# plot HMP last to be on top
pl.plot(MPTheta/(2*pi), MPPhi/(2*pi),'o-b',ms=ms)
pl.plot(HMPTheta/(2*pi), HMPPhi/(2*pi),'o-r',ms=ms)

phH = -1.386
th = np.arctan(float(N)/M)
print(eff(th, phH))

pl.show()
mrange=range(-4,5)
print("      N\M"),
for m in mrange: 
    if m!=0: print("{m:8d}".format(m=m)),
for n in range(-3,4):
    print("\n {n:8d}".format(n=n)),
    for m in mrange:
        if m!=0: print("{e:8.2f}".format(e=-eff(np.arctan2(n,m),phH))),


"""  2,5  0.53 
     2,6  0.440
     N,0  1
     1,3 about 1/3

reduction in apparent M = sin(2pi-th-ph)/cos(th) where ph = arctan(average(diff(HMPTheta)/diff(HMPPhi)))
= -1.386
and th = arctan(-N/M)

      N\M       -4       -3       -2       -1        1        2        3        4 
       -3     0.86     0.81     0.72     0.44     1.56     1.28     1.19     1.14 
       -2     0.91     0.88     0.81     0.63     1.37     1.19     1.12     1.09 
       -1     0.95     0.94     0.91     0.81     1.19     1.09     1.06     1.05 
        0     1.00     1.00     1.00     1.00     1.00     1.00     1.00     1.00 
        1     1.05     1.06     1.09     1.19     0.81     0.91     0.94     0.95 
        2     1.09     1.12     1.19     1.37     0.63     0.81     0.88     0.91 
        3     1.14     1.19     1.28     1.56     0.44     0.72     0.81     0.86

"""
