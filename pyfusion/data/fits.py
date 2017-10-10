""" gaussian and bi/mirrored exponential fits, excluding Nan data """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import leastsq

def gaus(x):  
    """ mis-spell to avoid clash """
    return(norm.pdf(x)/norm.pdf(0))

def mygauss(x, params):
    """ Scaled (c) , offset(b) Gaussian on pedestal (d) """
    a, b, c, d = params
    return(c * gaus(a*x + b) + d)

def mymirexp(x, params):
    """ mirrored exponential """
    a, b, c = params
    return(b * np.exp(-a*np.abs(x)) + c)

def mybiexp(x, params):
    """ bi- (independent) exponentials """
    a = params[0:2]
    b = params[2:4]
    c = params[4:6]
    i = (x>0).astype(int)
    return(b[i] * np.exp(-a[i]*np.abs(x)) + c[i])

def residuals(params, fn, x, y, yerrs=None):
    # add a general function, passed in args
    diffs = y - fn(x, params)
    if yerrs is not None:
        diffs = diffs/yerrs
    return(diffs)

def fitgauss(x_data, y_data, yerrs=None, ax=None, guess=None, pltkwargs={}):
    wnot = np.where(~np.isnan(y_data))[0]
    x = x_data[wnot]
    y = y_data[wnot]
    guess =  [.15, 0, 24, 10] if guess is None else guess
    newfit, cond = leastsq(residuals, guess, args=(x, y, mygauss))

    if ax is not None:
        xplot = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xplot, mygauss(xplot, newfit), **pltkwargs)
    if cond not in [1,2,3,4]: print('Failure in fits.py')
    return(newfit, cond, mygauss(x_data, newfit))


def fitbiexp(x_data, y_data, yerrs=None, ax=None, guess=None, pltkwargs={}):
    wnot = np.where(~np.isnan(y_data))[0]
    x = x_data[wnot]
    y = y_data[wnot]
    if yerrs is not None:
        yerrs = yerrs[wnot]
    guess =  [.15, .15, 30, 30, 24, 24] if guess is None else guess
    newfit, cond = leastsq(residuals, guess, args=(mybiexp, x, y, yerrs))

    if ax is not None:
        xplot = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xplot, mybiexp(xplot, newfit), **pltkwargs)
    if cond not in [1,2,3,4]: print('Failure in fits.py')
    return(newfit, cond, mybiexp(x_data, newfit))

def fitmirexp(x_data, y_data, yerrs=None, ax=None, guess=None, pltkwargs={}):
    wnot = np.where(~np.isnan(y_data))[0]
    x = x_data[wnot]
    y = y_data[wnot]
    if yerrs is not None:
        yerrs = yerrs[wnot]

    guess =  [.15, 30, 24] if guess is None else guess
    newfit, cond = leastsq(residuals, guess, args=(mymirexp, x, y, yerrs))

    if ax is not None:
        xplot = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xplot, mymirexp(xplot, newfit), **pltkwargs)
    if cond not in [1,2,3,4]: print('Failure in fits.py')
    return(newfit, cond, mymirexp(x_data, newfit))

if __name__ == '__main__':

    npts = 200 ; offs = 1.0; wid = 2.0
    # error ~ 1% for 200 pts, ~ 0.2% for 2000, ~0.02% for 20,000
    np.random.seed(0)
    data_x = np.linspace(-8, 10, npts) + np.random.rand(npts)/50
    data_y = gaus(data_x/wid + offs) + np.random.rand(npts)/10

    plt.plot(data_x, data_y)
    plt.title('fit to 1/width {w}, offs {o}'.format(w=1/wid, o=offs))

    guess = [1, 1, 1, 1]
    for tol in [1.0, 1e-3, 1e-6]:
        rets = leastsq(residuals, guess, args=(mygauss, data_x, data_y),
                       ftol=tol, full_output=True)
        newfit = rets[0]
        fit = str(rets[2]['nfev']) + ' iters, ' + ', '.join(
            ['{:.5f}'.format(parm) for parm in newfit])
        plt.plot(data_x, mygauss(data_x, newfit), label=fit)
        cond = rets[4]
        if cond not in [1,2,3,4]: print('Failure in fits.py')

    plt.legend()
    plt.show(0)
