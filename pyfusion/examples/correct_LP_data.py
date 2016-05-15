""" By looking at the params, rename the .npz file to the correct name
If done in-place, then the filenames must be sorted to avoid overwriting LP13

Note that the reading (and consistency check) of the pyfusion.cfg file is through 
the pyfusion interface.
This is different to the text file method in modify_cfg.py, and is probably more reliable,

"""
import numpy as np
import os
import pyfusion
import glob

sections = pyfusion.config.sections()
params_cfg = []
params_section = []

for section in sections:
    toks = section.split(':')
    if len(toks) != 2:
        continue
    typ, name = toks
    if not pyfusion.config.pf_has_option(typ, name, 'params'):
        continue
    params_section.append(name)
    param_str = pyfusion.config.pf_get(typ, name, 'params')
    params_cfg.append(eval('dict({ps})'.format(ps=param_str)))

def correctly_name(fn=None):
    # this is U or V for the files of interest (LP1_I0 will appear as a dup error otherwise
    unit = fn.split('_')[-1].split('.npz')[0]
    dat = np.load(fn)
    all_params_npz = dat['params'].tolist()

    params_npz = {}
    for key in 'DMD,ch,CDS'.split(','):
        params_npz[key] = all_params_npz[key]

    matches = [i for i in range(len(params_cfg)) 
               if (params_npz == params_cfg[i]) and ( params_section[i].endswith(unit))]

    if len(matches) != 1:
        raise LookupError('{num} matches to {fn} {pz}: {m}'
                          .format(fn=fn, pz=params_npz,  num=len(matches),
                                  m=[params_section[i] for i in matches]))
    
    newfn = fn.split('W7X')[0] + params_section[matches[0]] + '.npz'
    print('filename of {fn} should be {newfn}'.format(fn=fn, newfn=newfn))

    if os.path.exists(newfn):
        if 'LP10' in newfn:
            print('skipping ' + newfn)
        else:
            raise LookupError("Can't rename {fn}: {newfn} exists"
                          .format(fn=fn, newfn=newfn))

    os.rename(fn, newfn)

                

"""
fn = '/data/datamining/local_data/extra_data/20160310_9_W7X_L53_LP12_I.npz'
correctly_name(fn)
"""

# the sort is essential so that 13 is renamed before 15
for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/temp/20160310_9_*')):
    correctly_name(fn)
