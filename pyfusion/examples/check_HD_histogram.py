""" This script constructs a uniform random distribution in NDim dimensions
and plots the std function of the Mode class to check if the density
distribution compensation is reasonable. 
"""

import numpy as np
import pylab as pl


_var_defaults="""
NDim=5
n_bins=50
thisSTD=0.2
fact=None
n_iters=10  # number of attempts to equalise expected count in each bin
pow=None
NRand=int(2e6)
first_std = 0.5
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

def fn(fact,x,pow):
    return(fact * x**(pow))

 # NDim-1 is best - sometimes -0.5 looks better if the dist fn falls off early
if pow is None: pow = NDim-1 

from pyfusion.clustering.modes import Mode
test_mode=Mode('allpis',0,0,np.pi*np.ones(NDim),thisSTD*np.ones(NDim))
dist = test_mode.std(2*np.pi*np.random.random((NRand,NDim)))

pl.subplot(1,2,1)
(cnts,bins,patches)  = pl.hist(dist, bins=n_bins,log=1)
x=np.linspace(0.5,20,40)

if fact is None:
    checkbin = len(bins)/4
    fact = cnts[checkbin]/fn(1, bins[checkbin], pow)

pl.semilogy(x, fn(fact,x,pow),linewidth=3)
maxsd = np.sqrt(np.max(test_mode.csd**2))
max_valid_s = 1.5/maxsd  # I would have thought Pi/maxsd
pl.semilogy([max_valid_s, max_valid_s],pl.ylim(),'r--',linewidth=2)
#if pl.isinteractive():
pl.show(block=0)

# Now try to adjust the bin sizes so uniform dist is constant count
bins = [0,float(first_std)]
while len(bins) < n_bins:
    bins.append(2*bins[-1] - bins[-2])
    for iter in range(n_iters):
        corrected_width = (
            (bins[-2]-bins[-3]) * 
            # this is ratio of the definite integrals of s**NDim-1
            # for the last two bins, converted from counts/unit s to to counts
            ((bins[-2]**NDim - bins[-3]**NDim)/(bins[-2] - bins[-3]))/
            ((bins[-1]**NDim - bins[-2]**NDim)/(bins[-1] - bins[-2])))
        # print(corrected_width)
        bins[-1]=bins[-2] + corrected_width

pl.subplot(1,2,2)
if np.min(dist)>np.max(bins):
    raise ValueError('no counts in the first {n} bins up to {lastbin},'
                     ' increase first bin size (first_std)'
                     .format(n=len(bins), lastbin=bins[-1]))
pl.hist(dist, bins=bins,log=1)
pl.semilogy([max_valid_s, max_valid_s],pl.ylim(),'r--',linewidth=2)
pl.show(block=0)
