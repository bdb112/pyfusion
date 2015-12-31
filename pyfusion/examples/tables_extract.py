""" extract selected shots from a pytables hdf5 file 
Algorithm should be applicable for any 'on disk' variable format
"""

import pyfusion
from pyfusion.data.DA_datamining import DA

import numpy as np

dd = DA('$DA/DAHJ59k.npz',load=1).copyda()

shots = dd['shot']
ushots = np.unique(shots)

np.random.seed(0) # ensure repeatability = 
# chose 10 randomly
myshots = ushots[np.random.uniform(len(ushots),size=10).astype(int)]

inds = []
for shot in myshots:
    inds.extend(np.where(shots == shot)[0])

#     1,2,3,4,6,7,8,
#diff 1,1,1,2,1,1,1
#ones 1,1,1,0,1,1,1
#ups        3 
#downs    2, 

# sort in the order of the table - not necessarily shot order
inds = np.sort(inds)
ones = (np.diff(inds) == 1).astype(int)
# always take the first
ups = [-1]
ups.extend(np.where(np.diff(ones)==1)[0].tolist())
downs = np.where(np.diff(ones)==-1)[0].tolist()
downs.append(len(inds)+1)


#rg[i] [ups[i-1]+1:downs[i]]  for i=0..

test = []
for i in range(len(downs)):
    shts = shots[inds[ups[i]+1:downs[i]+2]]
    test.extend(shts)  # see comment below - real code probably should
    # use predifined arrays
    print(shts)

if len(test) != len(inds):
    print('inconsistent number of shots')
else:
    if np.all(test == shots[inds]):
        print('equal element by element')
    else:
        print('not equal element by element')

# now need to create the right sized arrays and selectively put elts
# in - this should be faster than appending when the data sizes
# approach memory limits.
