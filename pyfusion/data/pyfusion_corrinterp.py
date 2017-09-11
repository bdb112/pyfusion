import pyfusion as pf
from .corrinterp import correlation as plain_corrint
import numpy as np


def correlation(x, y, tx=None, ty=None, AC=True, coefft=True, vsfreq={}):
    """ vsfreq if not empty selects spectrum instead of a value - so far only for single variables.
    """
    if hasattr(x, 'timebase'):
        if tx is not None:
            print('Overriding timebase on ' +  x.channels.name if hasattr(x, 'channels') else '?')
        tx = x.timebase

    if hasattr(y, 'timebase'):
        if ty is not None:
            print('Overriding timebase on ' +  y.channels.name if hasattr(y, 'channels') else '?')
        ty = y.timebase

    if not hasattr(x, 'keys') or len(x.keys()) == 1:
        return [plain_corrint(y[ch], x.signal, ty, tx, AC=AC, coefft=coefft, vsfreq=vsfreq) for ch in y.keys()]
    elif not hasattr(y, 'keys'):
        return [plain_corrint(x[ch], y, tx, ty, AC=AC, coefft=coefft) for ch in x.keys()]
    else:
        corr2d = [[plain_corrint(x[xk], y[yk], tx, ty, AC=AC, coefft=coefft) for xk in x.keys()] for yk in y.keys()]
        shcoor2d = np.shape(corr2d)
        newshape = [shcoor2d[-1]]
        if np.product(shcoor2d[:-1]) == 1:
            return np.reshape(corr2d, newshape)
        elif np.sort(shcoor2d)[-2] == 1:
            return np.reshape(corr2d, [np.product(shcoor2d[:-1])] + newshape)
        else: return(corr2d)
