""" flip the phase of flucstructs according to the magnetic field direction
checks against phorig to make sure that phase is not flipped twice.
  works on dictionary of arrays "dd"
quick check:
(dd['phorig']*dd['b_0']*dd['phases'][:,0] >= 0).all() # True if with B

_PYFUSION_TEST_@@PRE@from pyfusion.data.DA_datamining import da ; dd=da()

"""


_var_defaults="""

inds = None  #  if None, do it for all instances
method = 'asB'  # asB, invertB, orig, flip, check

# o_copy is a copy of the original phases, the original of which
# is never touched.  ocopy is manipulated here so that cross checking 
# can be performed.

"""
import numpy as np

exec(_var_defaults)

from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

try:
    print('Using dd file from {fn}'.format(fn=dd.zip.filename))
except:
    print('could not find original filename')

wnan=np.where(np.isnan(dd['b_0']))[0];len(wnan)
wzero=np.where((dd['b_0']==0))[0];len(wzero)
wneg=np.where((dd['b_0']<0))[0];len(wneg)
wpos=np.where((dd['b_0']>0))[0];len(wpos)
print('negative:{wn:,}; zero:{wz:,}; positive:{wp:,}; NaN:{wnan:,};'
      .format(wn=len(wneg), wz=len(wzero), wp=len(wpos), wnan=len(wnan))),
if (len(wneg)+len(wpos)+len(wnan)+len(wzero)!=len(dd['b_0'])):
    raise ValueError("above do not add up to total of {t:,}".format(t=len(dd['b_0'])))
else:
    print('  => total consistent')


o_copy = 1*dd['phorig']
if (o_copy*dd['phases'][:,0] >= 0).all():
    print('appears to be untouched')
else: 
    print('appears to be altered in some way according to phorig')

if inds is None:  inds = np.arange(len(dd['b_0']))
whneg = np.where(dd['b_0'] < 0)[0]

if method == 'asB': # flip phase sign when B neg
    dd['phases'][whneg] = -dd['phases'][whneg]
    o_copy[whneg] = -o_copy[whneg]
elif method == 'invertB':   # flip phase ONLY for positive B
    dd['phases'][whneg,:] = -dd['phases'][whneg,:]
    dd['phases'][:,:] = -dd['phases'][:,:]
    o_copy[whneg] = -o_copy[whneg]
    o_copy[:] = -o_copy[:]
elif method == 'flip':
    print('not checking - just flipping')
    dd['phases'][:][:] = -dd['phases'][:][:]
elif method == 'check':
    print('checking - doing nothing')
else:
    raise ValueError('{m} is not a known method'.format(m=method))

if (o_copy*dd['phases'][:,0] >= 0).all():
    print('{m} OK!'.format(m=method))
elif (dd['phorig']*dd['b_0']*dd['phases'][:,0] <= 0).all():
    print('phase sign consistent with invertB')
else:
    print('Error - phase sign not consistent with B')

