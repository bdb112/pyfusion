import glob, os, shutil
import numpy as np
import logging
import time as tm

# see https://docs.python.org/2/howto/logging-cookbook.html
"""
logging.basicConfig(level=logging.DEBUG,filename='file_sorter.log', filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
#logging.getLogger('').addHandler(console)
"""
import pickle
#pickle.dump(ftable,open('ftable.pickle','wb'),protocol=2)  # 2 is python 2 compatible


from pyfusion.data.shot_range import shot_range

"""
# this is now the default pyfusion
# remove the pyfusion console logger
oldhandlers = logging.getLogger().handlers
if len(oldhandlers)>0: logging.getLogger().removeHandler(oldhandlers[0])

logging.basicConfig(filename='/tmp/file_sorter.log',level=logging.DEBUG,filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
"""

#ftable = pickle.load(open('ftable.pickle','rb'))
ftable = pickle.load(open('ftablePassPort4.pickle','rb'))
dirpath = "/tmp"
#dirpath = '/data/datamining/local_data/extra_data/may22'
dirpath = '/data/datamining/local_data/extra_data/tmp/'


logging.info('ftable has {l} entries, {tot} files in total'
             .format(l=len(ftable.keys()),tot=np.sum([len(lst) for lst in ftable.values()])))

# doesn't make sense to check dirpath for existing files - I don't know which subdir to look in

datelist = []
#for shot in shot_range([20160218,1], [20160310,1]):
#for shot in shot_range([20160218,1], [20160310,100]):
#for shot in shot_range([20160101,1], [20160218,1]):
#for shot in shot_range([20160218,1], [20160218,100]):
#for shot in shot_range([20160309,1], [20160309,100]):
for shot in shot_range([20160303,1], [20160303,100]):
    short = str(shot[0])
    if short not in datelist:
        datelist.append(short)

for date in datelist:
    fulldir = os.path.join(dirpath,date[-4:])
    if not os.path.isdir(fulldir):
        os.mkdir(fulldir)

    files = [k for k in ftable if date in k]
    for fil in files:
        paths = ftable[fil]
        ages = [os.path.getmtime(path) for path in paths]
        agesord = np.argsort(ages)
        newest = paths[agesord[-1]]
        if len(paths)>1:
            logging.debug('{tn}: {n} chosen from {num} over {t2}'
                          .format(tn=tm.ctime(ages[agesord[-1]]), 
                                  t2=tm.ctime(ages[agesord[-2]]),
                                  n=newest, num=len(paths)))

        fullpath = os.path.join(fulldir,os.path.split(newest)[1])
        if os.path.exists(fullpath):
            raise LookupError('file already present in the target path  {f}'
                              .format(f = fullpath))
        shutil.copy2(newest, fulldir)

"""

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

"""
