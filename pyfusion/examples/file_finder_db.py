""" find all W7X npz files and put all copies in a dictionary keyd by filename only.
"""
import glob, os
import numpy as np

ftable = {}
wild = '/data/datamining/local_data/extra_data/**'
wild = '/media/bdb112/PassportBDB4/**'
# probably don't need recursive=True - just use glob2/glob with **/201*npz

for fullfn in glob.glob(wild, recursive=True):  # only in 3.5 onwards
    if os.path.isdir(fullfn):
        continue
    nameext = os.path.split(fullfn)[1]
    name, ext = os.path.splitext(nameext)
    # print(name, ext)
    if ext != '.npz':
        continue
    if not name.startswith('201'):
        continue
    # now we hopefully only have W7X npz diagnostic data files
    if name not in ftable:
        ftable[name] = [fullfn]
    else:
        ftable[name].append(fullfn)

print('ftable has {l} entries, {tot} files in total'.
      format(l=len(ftable.keys()),tot=np.sum([len(lst) for lst in ftable.values()])))

import pickle
pickle.dump(ftable,open('ftable_new.pickle','wb'),protocol=2)  # 2 is python 2 compatible
# pickle.load(open('ftable.pickle','rb'))
