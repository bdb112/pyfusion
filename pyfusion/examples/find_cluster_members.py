""" Given a set of lists of angles, find those members closest to other members
   Simple minded but slow algorithm is:
  1/ Start with a 'centre' list, which may be given, or defaults to the mean f all the lists mapped to a certain rage of 2pi
  2/ successively delete the furthest from the current 'centre' and recalculate the 'centre' until the end criterion is met:
   Either the maximum mean error meets a criterion or the a certain fraction of the members remain
   Not clear how to choose the 'centre' using means_

A true clusterer would be better
"""
import numpy as np
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden

def dist_wrap(l1, l2, norm_index=2):
    """ return the average distance modulo 2pi between two lists of angles
    The average is calculated as an n-norm.
    >>> round(dist_wrap([2, 1, 0], [-4, 1, 3]), 7)
    3.013336

    """
    diffs = modtwopi(np.array(l1) - np.array(l2), offset=0)
    return np.linalg.norm(diffs, norm_index)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Slow: around 1000 13 element lists/second - probably could eliminate
#  more than one at a time.  However it may make more sense to look for
#  clusters - this will work if there are two types of shapes.
    
from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file


min_left = 100
tolerance = 0.1


da = DA('DAMIRNOV_41_13_15_3m_20180808_5.npz', load=1)
#da = DA('DAMIRNOV_41_13_BEST_LOOP_10_3ms_20180912043.npz', load=1)
wh = da['indx']
#wh = np.where((btw(da['freq'],2,4)) & (da['amp']>0.012) )[0]
phs = da['phases'][wh].tolist()
inds = da['indx'][wh].tolist()

ctrs = np.mean(modtwopi(phs, 0), 0)
num = len(ctrs)

for left in range(len(phs), min_left, -1):
    dists = [dist_wrap(ctrs, phases)/num for phases in phs]
    if np.max(dists) < tolerance:
        break
    worst = np.argmax(dists)
    phs.pop(worst)
    inds.pop(worst)

print('{num} found, average scatter is {avg:.2f} rad'.format(num=len(inds), avg=np.average(dists)))
