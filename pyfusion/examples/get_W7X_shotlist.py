""" 
Find which shots are complete for a given diagnostic.
Find how many channels/shots are missing? 

For a diagnostic, 
 1/ find which shots are fully cached
 2/ find which shots are close to cached

Look for all cached MIRNOV shots

opts = pyfusion.config.pf_options('Diagnostic', 'W7X_MIRNOV_41_3')
pyfusion.config.pf_get('Diagnostic','W7X_MIRNOV_41_3','channel_0')

"""

import pyfusion
import numpy as np


diag_name = 'W7X_MIRNOV_41_BEST_LOOP'
opts = pyfusion.config.pf_options('Diagnostic', diag_name)  # channels are amongst options
chans = [pyfusion.config.pf_get('Diagnostic', diag_name, opt)
         for opt in opts if 'channel_' in opt]

names = np.loadtxt('W7X_MIR/allmirhomeL', dtype='S', delimiter='#$%').tolist()
names += np.loadtxt('W7X_MIR/allmirfreiL', dtype='S', delimiter='#$%').tolist()

common = []
for chan in chans:
    matches  = [nm.split('/')[-1].split('_')[:2] for nm in names if chan in nm and 'short' not in nm]
    print(chan, len(matches))
    if len(common) == 0:
        common = matches
    else:
        common = [shot for shot in common if shot in matches]
        
# Note: np.sort wrecks this - the shots and date are mixed up??
common_shots = [tuple([int(st) for st in shot]) for shot in common]
