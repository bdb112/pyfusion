"""
_PYFUSION_TEST_@@PRE@from pyfusion.data.DA_datamining import da ; myDA=da(dd=False) ; myDA.extract(locals())
"""
import numpy as np
import pylab as pl
n_phases = len(phases[0])
(f, axes) = pl.subplots(n_phases, sharex=True)
f.subplots_adjust(hspace=0)
pl.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

for (i,ax) in enumerate(axes.flatten()):
    ax.hist(phases[:,i],np.linspace(-3.16,3.16,316))
    pl.ylabel('frequency')
pl.show(block=0)

"""
import pyfusion
chans = []
diag = 'MP2010HMPno612'
pyfusion.config.get('Diagnostic:HMPno612', 'channel_1')
for 
"""
