import pyfusion
import numpy as np
import pylab as pl
from pyfusion.utils import fix2pi_skips, modtwopi
diag_name = 'VSL_SMALL'
diag_name = 'MP'

dev=pyfusion.getDevice("LHD")
data=dev.acq.getdata(27233,diag_name)
#data.plot_signals()
# extract the phi DIrectly from the cfg file
Phi = np.array([2*np.pi/360*float(pyfusion.config.get
                                  ('Diagnostic:{cn}'.
                                   format(cn=c.name), 
                                   'Coords_reduced')
                                  .split(',')[0]) 
                for c in data.channels])

pl.subplot(121)
pl.plot(Phi, linestyle='steps')
Phi_circ = fix2pi_skips(np.append(Phi, Phi[0]), around=0)
dp = np.diff(Phi_circ)
pl.plot(dp, linestyle='steps')
pl.show()
pl.subplot(122)
phases = []
for N in range(-5, 5):
    ph = modtwopi(N*dp, offset=0)
    phases.append(ph)
    pl.plot(phases[-1], label='N={N}'.format(N=N))
pl.legend()
pl.show()

# fake up a clusters file
clinds = [[c] for c in np.arange(len(phases))]
subset = phases
subset_counts = np.ones(len(phases))
np.savez_compressed('ideal_toroidal_modes', clinds=clinds, subset=subset, subset_counts=subset_counts, dphase=-1, sel=np.arange(10, 16))
