#faster version of dist - with option of omitting sqrt
# uses only float32 - better speed than 16, less memory than 64
from __future__ import division

# slight tweak to improve speed by 10%
#gcc -shared -pthread -fPIC -fwrapv -O5 -mhard-float -Wall -fno-strict-aliasing       -I/usr/include/python2.7 -o dist.so dist.c

cdef extern from "math.h":
    float fmodf(float, float) nogil
    float sqrtf(float) nogil

cdef float pi=3.141592653589793

def simple_dist(float x, float y):
    return(fmodf((x-y)**2,2*pi))

# this is based on docs.cython.org/src/tutorial/numpy.html

import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# FTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
FTYPE = np.float32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t FTYPE_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
#
# The arrays x, y and h is typed as "np.ndarray" instances. The only effect
# this has is to a) insert checks that the function arguments really are
# NumPy arrays, and b) make some attribute access like f.shape[0] much
# more efficient. (In this example this doesn't matter though.)

#def dist(np.ndarray x, np.ndarray y, int squared=0, int debug=0):
# using [FTYPE_t, ndim=1] etc cut the time from 8us to 6.2ns
# 8ns (10,1e6) for squared, no fmod. - extra
def dist(np.ndarray[FTYPE_t, ndim=1] x, np.ndarray[FTYPE_t, ndim=2] y, squared=None, averaged = None, int debug=0):
    if x.shape[0] != y.shape[1]:
       raise ValueError("must have same length")

    cdef int c_average
    cdef int c_squared

    # doesn't matter how fast averaged is - only used once.
    if averaged is None:
        c_averaged = 1
    else:        
        c_averaged = averaged

    # speed of access to squared is more important
    if squared is None:
        c_squared = 1
    else:        
        c_squared = squared
   
    assert x.dtype == FTYPE and y.dtype == FTYPE        
    cdef int slen = x.shape[0] # short length
    cdef int llen = y.shape[0] # long length - returns garbage if 1D?
    cdef np.ndarray[FTYPE_t, ndim=1] d = np.zeros([llen], dtype=FTYPE)

    cdef int i_s
    cdef int i_l

    cdef float s
    cdef float fact

    if c_averaged: 
        fact = 1/slen
    else:
        fact = 1

    if debug: print(slen, llen)

    with nogil:
        for i_l in range(llen):
            s = 0.
            for i_s in range(slen):
                # s += (x[j] - y[j,i])**2  # 8ns
                s += (fmodf(x[i_s] - y[i_l,i_s]+5*pi,2*pi)-pi)**2 # 26ns - all fmodf
                                                   # 1.5 ns w -O5 instead of O2
            if c_squared: 
                d[i_l] = s*fact
            else:
                d[i_l] = sqrtf(s*fact)

    return(d)

"""
#test code
#import dist
import pyximport; pyximport.install()
from pyfusion.utils.dist_nogil import dist
v14 = dist.dist(np.arange(14,dtype=np.float32)/5.,
                np.ones((10,14),dtype=np.float32),squared=1, averaged=0)
vplus14 = dist.dist(20*np.pi+np.arange(14,dtype=np.float32)/5.,
                    np.ones((10,14),dtype=np.float32),squared=1, averaged=0)
vminus14 = dist.dist(-2*np.pi+np.arange(14,dtype=np.float32)/5.,
                     np.ones((10,14),dtype=np.float32),squared=1, averaged=0)
a14 = np.sum((np.arange(14)/5.-1)**2)

np.allclose(v14, a14, rtol=5e-7)
np.allclose(vplus14, a14, rtol=9e-6)
np.allclose(vminus14, a14, rtol=5e-7)
"""