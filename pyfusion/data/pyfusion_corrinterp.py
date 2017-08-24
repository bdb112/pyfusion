import pyfusion as pf
from .data import correlation as plain_corrint


def correlation(x, y, tx=None, ty=None, AC=True, coefft=True):
    if hasattr(x, 'timebase'):
        if tx is None:
            print('Overriding timebase on ' + x.name)
        tx = x.timebase

    if hasattr(y, 'timebase'):
        if ty is None:
            print('Overriding timebase on ' + y.name)
        ty = y.timebase

    return plain_corrint(x, y, tx, ty, AC=AC, coefft=coefft)
