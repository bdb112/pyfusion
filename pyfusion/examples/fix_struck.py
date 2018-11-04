""" Rotate the order of digitisers in a 'struck' digitizer to correct bug, initially
only from the archive.  Won't work on stored npz data.
Example:
pyfusion.reload_config()
run pyfusion/examples/fix_struck.py 1
"""

import pyfusion
import sys

offset = 1 if len(sys.argv) == 1 else int(sys.argv[1])

# select out W7X mdsplus diags to narrow the field
W7Mdiags = [diag for diag in pyfusion.config.sections() if diag.startswith('Diagnostic:W7M')]
# select out signals
W7Msigs = [diag for diag in W7Mdiags if 'mds_path' in pyfusion.config.options(diag) and 'data_fetcher' in pyfusion.config.options(diag)]
# sub select digitizer channels
W7Mdigs = [diag for diag in W7Msigs if 'Multi' not in pyfusion.config.get(diag, 'data_fetcher')]
# then those in the same digitizer
MLPdiags = [[x, pyfusion.config.get(x, 'mds_path')] for x in W7Mdigs if 'SIS8300KU_1' in pyfusion.config.get(x, 'mds_path')]
for ch, mdsp in MLPdiags:
    # str((offset + int(mdsp.split('KU_')[1].split('.CH')[0])) % 10)
    pref, num = mdsp.split('.CHANNEL')
    newnum = str((offset + int(num)) % 10)
    newp = pref + '.CHANNEL' + newnum
    pyfusion.config.set(ch, 'mds_path', newp)

