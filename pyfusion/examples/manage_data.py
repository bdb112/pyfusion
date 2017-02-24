""" replaces file_sorter - looks in a bunch of paths for W7X files for a partial string match (target),
and picks the newest of those to copy to a destination,  warning if the newest is not the largest.
"""
from __future__ import print_function
import os, sys, pickle, shutil
import numpy as np
import pyfusion  # only to get usual pyfusion logging format
import logging

debug =1

def get_latest(paths):
    if len(paths) == 1:
        return(paths[0])
    else:
        times = [os.path.getmtime(p) for p in paths]
        sizes = [os.path.getsize(p) for p in paths]
        newest = np.argmax(times)
        if sizes[newest] == np.max(sizes):
            pass
        else:
            logging.warning('Newest file {fn} is {pc:.3f}% smaller than {fb}?? '
                            .format(pc=100 * (np.max(sizes)-sizes[newest])
                                    /float(np.max(sizes)),
                                    fn=paths[newest], fb=paths[np.argmax(sizes)]))
        return(paths[newest])

_var_defaults="""
search_root = '/media/bdb112/CE6A1E7D6A1E630F/W7X_OP1_1_original'
# on W7X Virt PC  need two back slashes on search_path='c:\\' 
search_paths = [search_root + '/**']
to_root = '/data/datamining/local_data/W7X/'
##to_root = '/tmp/'
to_root = ''
this_dir = '0217'
# Done: 0308 0310 0309 217
target = '2016{md}_9_W7X_'.format(md=this_dir)
target = '2016{md}_'.format(md=this_dir)
# target = '2016{md}_1_W7X_L57'.format(md=this_dir)
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if sys.version < (3, 5, 0):
    try:
        import glob2 as glob
    except ImportError:
        print('Can`t access recursive glob - won`t work in subdirs')
        # recursive glob for glob 2 uses /**  or /**/*abc

try:
    allpaths
    if oldsearch_paths != search_paths:
        raise NameError
except NameError:
    allpaths = []
    for sp in search_paths:
        allpaths.extend(glob.glob(os.path.join(search_root, sp)))
    #allpaths = pickle.load(open('ftablePassPort4.pickle','rb'))

if len(allpaths) != 1:
    print('{n} files found in {search_paths}'
          .format(n=len(allpaths), search_paths=search_paths))

names = [os.path.split(f)[-1] for f in allpaths]
print('{u} unique names'.format(u=len(np.unique(names))))

oldsearch_paths = search_paths #  save for a repeat (run -i ) session
allmatches = [f for f in allpaths if target in os.path.split(f)[-1]]
names = np.unique([os.path.split(m)[-1] for m in allmatches])
directories = np.unique([os.path.split(am)[0] for am in allmatches])
wanted = [nm for nm in names if target in nm]
print('processing {n} unique files of {am} matching {tg}, in \n'#{dirs}'
      .format(n=len(wanted), am=len(allmatches), tg=target))
for dir in directories:
    c = len([d for d in allmatches if dir == os.path.split(d)[0]])
    print('{c:10}: {dir}'.format(c=c, dir=dir))

if to_root == '':
    sys.exit()
to_path = os.path.join(to_root, this_dir)
if not os.path.isdir(to_path):
    os.mkdir(to_path)

for (w, nm) in enumerate(wanted):
    paths = [p for p in allpaths if nm in os.path.split(p)[-1]]
    if len(paths)>1:
        chosen = get_latest(paths)
    else:
        chosen = paths[0]
    logging.info('{nm}: {fc} of {n} {paths}'
                 .format(nm=nm, fc=chosen.replace('search_root',''), n=len(paths), 
                         paths=[p.replace(search_root,'').replace(nm,'') 
                                for p in paths]))

    full_name = os.path.join(to_path, nm)
    if os.path.exists(full_name):
        raise LookupError('file already present in the target path  {f}'
                          .format(f = full_name))
    else:
        print('.', end=['','\n'][w%80==0])
        shutil.copy2(chosen, os.path.split(full_name)[0]) # safer than using to_path

print('logged to', [lh.stream.name for lh in logging.getLogger().handlers])
