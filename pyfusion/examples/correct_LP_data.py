""" By looking at the params, rename the .npz file to the correct name
If done in-place, then the filenames must be sorted in the glob argument 
to avoid overwriting LP13.

Note that the reading (and consistency check) of the pyfusion.cfg file is through 
the pyfusion interface.
This is different to the text file method in modify_cfg.py, and is probably more reliable,
#_PYFUSION_TEST_@@Skip
"""
import numpy as np
import os
import pyfusion
import glob

sections = pyfusion.config.sections()
params_cfg = []
params_section = []
verbose = 0
dry=0
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
    if unit not in ['U','I']:
        return
    dat = np.load(fn)
    all_params_npz = dat['params'].tolist()

    params_npz = {}
    try:
        for key in 'DMD,ch,CDS'.split(','):
            params_npz[key] = all_params_npz[key]
    except KeyError as reason:
        msg = str('Missing DMD params - moving to {fn}.old \n {r}'.format(fn=fn, r=reason))
        pyfusion.logging.error(msg)
        if not dry: 
            os.rename(fn, fn+'.old')
        return

    matches = [i for i in range(len(params_cfg)) 
               if (params_npz == params_cfg[i]) and ( params_section[i].endswith(unit))
               and 'W7XM' not in params_section[i] ]

    if len(matches) != 1:
        raise LookupError('{num} matches to {fn} {pz}: {m}'
                          .format(fn=fn, pz=params_npz,  num=len(matches),
                                  m=[params_section[i] for i in matches]))
    
    newfn = fn.split('W7X_L')[0] + params_section[matches[0]] + '.npz'
    if verbose>1: print('filename of {fn} should be {newfn}'.format(fn=fn, newfn=newfn))

    if os.path.exists(newfn):
        if 'LP10' in newfn:
            pyfusion.logging.info('skipping ' + newfn)
        else:
            msg = str("Can't rename {fn}: {newfn} exists"
                      .format(fn=fn, newfn=newfn))
            pyfusion.logging.error(msg)
            raise LookupError(msg)
    else:
        pyfusion.logging.debug(str('renaming {fn} to {newfn}'.format(fn=fn, newfn=newfn)))
        os.rename(fn, newfn)
    return([fn, newfn])
                

"""
fn = '/data/datamining/local_data/extra_data/20160310_9_W7X_L53_LP12_I.npz'
correctly_name(fn)
"""
goods = []
# the sort is essential so that 13 is renamed before 15
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/temp/20160310_39_*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may2016/0224/20160224_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0224/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/tmp/0218/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0310/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0309/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/tmp/0309/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0303/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0302/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0301/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0223/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0210/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0217/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0204/x/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0203/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0202/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0209/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0128/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0127/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0126/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0122/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0121/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0120/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/extra_data/may22/0119/2016*_*LP*')):
# second go in Jan 2017
#for fn in np.sort(glob.glob('/data/datamining/local_data/W7X/0308/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/W7X/0310/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/W7X/0309/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/W7X/0217/2016*_*LP*')):
#for fn in np.sort(glob.glob('/tmp/0309/2016*_*LP*')):
#for fn in np.sort(glob.glob('/data/datamining/local_data/W7X/0224/2016*_*LP*')):
#for fn in np.sort(glob.glob('/tmp/0217/2016*_*LP*')):
for fn in np.sort(glob.glob('/data/datamining/local_data/W7X/0218/2016*_*LP*')):
    goods.append(correctly_name(fn))
print('renamed {l} files - see goods'.format(l=len(goods)))
