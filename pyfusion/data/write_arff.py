# for a dict of arrays (dd), all same array size, write an arff file, converting complex data to magnitude and phase
""" for complex, had to unroll the loop to write a line of data - slows down about 7x!!
"""

import os

import numpy as np
from numpy.random.mtrand import uniform 

# first, arrange a debug one way or another
try: 
    from debug_ import debug_
except:
    def debug_(debug, msg='', *args, **kwargs):
        print('attempt to debug ' + msg + 
              " need boyd's debug_.py to debug properly")
debug=1

# then generate data if there is none
try:
    dd.keys()
    print('using existing dd dataset')

except:
    print('making dummy data')
    relation = 'test'
    dd = {'f': uniform(size=(10)), 'i': range(10), 
          'npf':np.linspace(0,10,10), 
          'c64': np.exp(1j*np.linspace(0,10,10000))}#,'s': ['a','bb']}

def tomagphase(dd, key):
    """ This would be happier as class method
    given a dictionary of arrays dd, and a key to complex data,
    add key and data for the magnitude and another for the phase, then
    remove the original key.

    This destructive code has the advantage that the write is sped up by
    seven times.
    """
    dd.update({'mag_'+key: np.abs(dd[key])})
    dd.update({'phase_'+key: np.angle(dd[key])})
    dd.pop(key)

def split_vectors(dd):   #  , keep_scalars=False):
    """ pop any vectors and replace them with scalars consecutively numbered
    """
    sub_list = []
    for k in list(dd.keys()):
        order = len(np.shape(dd[k]))
        if order==0:
            dd.pop(k)
        elif order>1:
            cols = np.array(dd.pop(k)).T
            newks = []  # a list of vector keys and the corresponding split ones
            for i in range(len(cols)):
                newk = '{k}_{i}'.format(k=k, i=i)
                dd.update({newk: cols[i]})
                newks.append(newk)
            sub_list.append([k, newks])

        else:
            pass # keep  column vectors
    return(sub_list)

            
def write_arff(da, filename='tmp.arff', use_keys=[]):
    """ use_keys - keys to save, empty list means all
    """
    if os.path.exists(filename):os.unlink(filename)

    f = open(filename,'w')
    dd = da.copyda()  # need to copy before vectors are split

    # now replace any keys in the key list with their split names
    if len(use_keys) == 0:   # if empty list, use all
        ks = np.sort(dd.keys()).tolist()
    else:
        ks = use_keys

    if 'info' in ks: ks.remove('info') # arff can do info struct.

    sub_list = split_vectors(dd)
    for key,newks in sub_list: 
        ks.remove(key)
        ks.extend(newks)

    f.write("@relation {0}\n".format(relation))
    for k in ks:
        f.write("@attribute {0} numeric\n".format(k))

    f.write('@data'+"\n")
    inds = range(len(dd[ks[0]]))
    for ind in inds:
        f.write(','.join([str(dd[k][ind]) for k in ks]) + "\n")

    f.close()

if __name__ == '__main__':

    # original primitive code - save the dict in dd
    FILE_NAME = 'tmp.arff'
    if os.path.exists(FILE_NAME):os.unlink(FILE_NAME)

    f = open(FILE_NAME,'w')
    ks = np.sort(dd.keys())
    f.write("@relation {0}\n".format(relation))
    for k in ks:
        f.write("@attribute {0} numeric\n".format(k))

    f.write('@data'+"\n")
    inds = range(len(dd[ks[0]]))
    for ind in inds:
        f.write(','.join([str(dd[k][ind]) for k in ks]) + "\n")

    f.close()

"""
sep14 = nc_storage('sep14.nc')
dd=sep14.restore()
wt=where(abs(dd['time'] -30) < 5)[0]
for k in dd.keys(): exec("dd['{0}']=array(dd['{0}'][wt])".format(k))
x=dd.pop('ne_array')
x=dd.pop('phases')
for coil in 'xyz': tomagphase(dd,'coil_1'+coil)
run -i write_arff.py
"""
