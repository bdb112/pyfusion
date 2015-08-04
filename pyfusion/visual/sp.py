import numpy as np
import pylab as pl
import matplotlib
import pyfusion
from pyfusion.utils.utils import decimate

from pyfusion.debug_ import debug_

debug=0

def size_val(marker_size, size_scale, dot_size):
    if size_scale<0: 
        return(-size_scale*np.exp(np.sqrt(marker_size/(dot_size/20))))
    else:
        return(size_scale*(np.sqrt(marker_size/dot_size)))
 

def sp(ds, x=None, y=None, sz=None, col=None, c=None, decimate=0, ind = None, nomode=None,
       size_scale=None, dot_size=30, hold=0, seed=None, colorbar=None, 
       dither=0, legend=True, marker='o',**kwargs):
    """ Scatter plot front end, size_scale 
    x, y, sz, col can be keys or variables (of matching size to ds)
    decimate = 0.1 selects 10% of the input, -0.1 uses a fixed key. 
    Note: This code is messy, but the idea is sound - that is
    1: Exploit the flexible colour and size of scatter
    2: flexible inputs via string names or by arrays
    3: Treat mode numbers specially to suppress "undefined" mode numbers
    dither 0: no effect
    -ve random seed,  +ve repeated seed
    c is the same as col for compatibility with satter
    returns a collection so that they can be selectively turned on an off
    """
    if col is not None and c is not None: 
        print('***conflicting values for c and col')
    elif col is None and c is not None:
        col = c
    if type(ds) == type({}): keys = np.sort(ds.keys())
    elif type(ds) == np.ndarray: keys = np.sort(ds.dtype.names)
    else: raise ValueError(
        'First argument must be a dictionary of arrays or an '
            'array read from loadtxt.  For DA() objects, remember to load!')

    if x is None: x = keys[0]        
    if y is None: y = keys[1]        
    if col is None: 
        if keys is None:
            col = None
        else:
            col = keys[2]        


    # deal with the indices (length) first, so we can consider indexing x,y earlier
    if ind is None: 
        if pl.is_string_like(x):
            lenx = len(ds[x])
        else: 
            lenx = len(x)
        ind = np.arange(lenx)

    if seed != None: np.random.seed(seed)
    if decimate != 0: 
        if decimate<0: np.random.seed(0)  # fixed seed for decimate<0
        ind = ind[(np.where(np.random.rand(len(ind))<abs(decimate)))[0]]
    else:  # decimate if very long array and decimate == 0
        if (len(ind) > 2e4):   # 2e5 for scatter, 1e5 for plot
            print('Decimating automatically as data length too long [{0}]'
                  .format(len(ind)))
            ind = ind[np.where(np.random.rand(len(ind))<(2e4/len(ind)))[0]]
    if len(ind) == 0:
        print('nothing to plot')
        return()

    if pl.is_string_like(x):
        x_string = x
        x = ds[x]
    else:
        x_string = ''

    if pl.is_string_like(y):
        y_string = y
        y = ds[y]
    else:
        y_string = ''

    size_string = '<size>'
    color_string = '<color>'

    if pl.is_string_like(col): 
        if np.any(np.array(keys)== col):  # AN ITEM?
            color_string = col
            col=ds[col]
        else: col = col  #  or a specific colour
    else:
        color_string = ''
        if col is None: 
            col='b'
    if len(np.shape(col))>0:   # an array or list
        col = np.array(col)[ind]
        if np.issubdtype(col[0], int):
            # if they are ints, assume mode numbers, and find nomode of required
            if nomode is None:
                if hasattr(col, 'dtype'):
                    col_dtype = col.dtype
                    minint = np.iinfo(col_dtype).min
                    nomode = minint
                else:
                    if len(col) != 0:
                        nomode = np.iinfo(col[0]).min
                    else:
                        nomode = np.iinfo(col).min

            w_not_nomode = np.where(nomode != col)[0]
            # shrink ind further to avoid displaying unidentified modes
            ind = ind[w_not_nomode]
            col = col[w_not_nomode]
        else:  # if real, scale to 0..256
                pass
            
    if dither != 0:  # 
        if dither>0:
            np.random.seed(0)
        xdither = dither*(np.max(x) - np.min(x))
        ydither = dither*(np.max(y) - np.min(y))

        x = x + xdither * (np.random.random(len(x))-0.5)
        y = y + ydither * (np.random.random(len(x))-0.5)

    if sz is None: sz=20 * np.ones(len(x))
    if pl.is_string_like(sz): 
        size_string = sz # size scale is the value giving a dot size of dot_size
        sz=ds[sz]
    else: # must be a number or an array
        sz = np.array(sz)

    if size_scale is None: size_scale = max(sz[ind])

    if size_scale<0:  # negative is a log scale
        sz=dot_size/20*(np.log(sz[ind]/-size_scale))**2
    else: 
        sz=dot_size*(sz[ind]/size_scale)  # squarung may make sense, but too big

        
    if max(sz)>1000: 
        if pl.isinteractive():
            inp=raw_input('huge circles, radius~ {szm:.3g}, Y/y to continue'
                          .format(szm = max(sz.astype(float)))) # bug! shouldn't need asfloat
            if inp.upper() != 'Y': raise ValueError
        else:
             warn('reducing symbol size')
             sz=200/max(sz) * sz
    debug_(debug,3)

    if hold==0: pl.clf()    
    coll = pl.scatter(x[ind],y[ind],sz,col, hold=hold,marker=marker,**kwargs)
#    pl.legend(coll   # can't select an element out of a CircleCollection
    sizes = coll.get_sizes()
    max_size=max(sizes)
    big=matplotlib.collections.CircleCollection([max_size])
    med=matplotlib.collections.CircleCollection([max_size/10])
    sml=matplotlib.collections.CircleCollection([max_size/100])
    if legend == True:
        pl.legend([big,med,sml],
                  [("%s=%.3g" % (size_string,size_val(max_size, size_scale, dot_size))),
                   ("%.3g" % (size_val(max_size/10,size_scale, dot_size))),
                   ("%.3g" % (size_val(max_size/100, size_scale, dot_size)))])

    pl.xlabel(x_string)
    pl.ylabel(y_string)
    pl.title('size=%s, colour=%s' % (size_string, color_string))
    if colorbar is None and len(col) > 1:
        colorbar = True
    if colorbar: pl.colorbar()
    return(coll)  # 
