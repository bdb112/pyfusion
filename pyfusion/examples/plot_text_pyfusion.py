"""
This has been largely supeseded by splitting into acquisition/read_text_pyfusion and visual/sp.py
Extract data from the text output of the new pyfusion code
And plot anything vs anything: (not implemented yet?)
 Default will plot freq vs t_mid for the first shot
In any case, can use the sp function easily from the command line
Just use the scatter function with the appropriate quantity as the 'index'
e.g    pl.scatter(ds['t_mid'][sind], freq_scale*ds['freq'][sind])
To reuse old data, use -i option on run (in ipython)

run ../../plot_text_pyfusion.py
run ../../plot_text_pyfusion.py shot=85000 skip=3

dot_size scales the disc up, amp_scale controls the log effect - signals around "amp_scale" are less compressed than others


sp(ds,'t_mid','freq',sz='p',col='amp',size_scale=.1,decimate=0.05,hold=0,ind=where(ds['amp']>0.1)[0])

"""
import numpy as np
from numpy import max
import pylab as pl
import os
import matplotlib
from matplotlib.cbook import is_string_like, is_numlike
from warnings import warn
from pyfusion.debug_ import debug_

debug=0

def extract_vars(datadict, varlist, indices=None, debug=debug):
    """ convenience routine to copy data from a dictionary (datadict) into separate variables.
    
    This can be used to sub-select different sized samples without affecting the
    main datastore.
    
    example: make arrays freq and t_mid as locals from entries in the dictionary
    ds, which was created by plot_text pyfusion.
    accepts varlist as a list or a dictionary (which allows renaming)

    locals().update(extract_vars(ds,['freq','t_mid']))
    locals().update(extract_vars(ds,{'freq':'newf','t_mid':'newtime'}))
    """
    vardict = {}
    if type(varlist) == type([]): # make a list into a trivial dictionary
        dict = {}
        for var in varlist: 
            dict.update({var: var})


    varlist = dict
    if indices != None:
	ind = '[indices]'
    else: 
	ind = ''

    for var in varlist.keys():
        cmd = "vardict.update({varlist[var]:datadict[var]" + ind + "})"
        if debug > 0: print(cmd)
        exec(cmd)
    return(vardict)

def sp(ds, x=None, y=None, sz=None, col=None, decimate=0, ind = None, 
       size_scale=None, hold=0, seed=None, colorbar=None, marker='o'):
    """ Scatter plot front end, size_scale 
    x, y, sz, col can be keys or variables (of matching size to ds)
    decimate = 0.1 selects 10% of the input, -0.1 uses a fixed key. 
    """
    def size_val(marker_size):
        if size_scale<0: 
            return(-size_scale*np.exp(np.sqrt(marker_size/(dot_size/20))))
        else:
            return(size_scale*(np.sqrt(marker_size/dot_size)))
 
    if type(ds) == type({}): keys = np.sort(ds.keys())
    elif type(ds) == np.ndarray: keys = np.sort(ds.dtype.names)
    else: raise ValueError(
        'First argument must be a dictionary of arrays or an '
            'array read from loadtxt')

    if x is None: x = keys[0]        
    if y is None: y = keys[1]        
    if col is None: col = keys[2]        


    # deal with the indices first, so we can consider indexing x,y earlier
    if ind is None: 
        if is_string_like(x):
            lenx = len(ds[x])
        else: 
            lenx = len(x)
        ind = arange(lenx)

    if seed != None: np.random.seed(seed)
    if decimate != 0: 
        if decimate<0: np.random.seed(0)  # fixed seed for decimate<0
        ind = ind[(where(np.random.rand(len(ind))<abs(decimate)))[0]]
    else:  # decimate if very long array and decimate == 0
        if (len(ind) > 1e5): 
            print('Decimating automatically as data length too long [{0}]'
                  .format(len(ind)))
            ind = ind[where(np.random.rand(len(ind))<(2e4/len(ind)))[0]]

    if is_string_like(x):
        x_string = x
        x = ds[x]
    else:
        x_string = ''

    if is_string_like(y):
        y_string = y
        y = ds[y]
    else:
        y_string = ''

    size_string = '<size>'
    color_string = '<color>'

    if is_string_like(col): 
        if np.any(np.array(keys)== col):
            color_string = col
            col=ds[col][ind]
        else: col = col  # colour is hardwired    
    else:
        if col is None: col='b'
        else:
            col = np.array(col)[ind]
        color_string = ''


    if sz is None: sz=20 * np.ones(len(x))
    if is_string_like(sz): 
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
            inp=raw_input('huge circles, radius~ {:.3g}, Y/y to continue'
                          .format(max(sz)))
            if inp.upper() != 'Y': raise ValueError
        else:
             warn('reducing symbol size')
             sz=200/max(sz) * sz
    debug_(debug,3)

    if hold==0: pl.clf()    
    coll = pl.scatter(x[ind],y[ind],sz,col, hold=hold,marker=marker,label='')
#    pl.legend(coll   # can't select an element out of a CircleCollection
    sizes = coll.get_sizes()
    max_size=max(sizes)
    big=matplotlib.collections.CircleCollection([max_size])
    med=matplotlib.collections.CircleCollection([max_size/10])
    sml=matplotlib.collections.CircleCollection([max_size/100])
    pl.legend([big,med,sml],
           [("%s=%.3g" % (size_string,size_val(max_size))),
            ("%.3g" % (size_val(max_size/10))),
            ("%.3g" % (size_val(max_size/100)))])

    pl.xlabel(x_string)
    pl.ylabel(y_string)
    pl.title('size=%s, colour=%s' % (size_string, color_string))
    if colorbar is None and len(col) > 1:
        colorbar = True
    if colorbar: pl.colorbar()

phase_array = np.zeros(5)
verbose=1
amp_scale = 1
freq_scale = 1
dot_size=30  
debug=True

#test_dtype = [ ('shot','i8'), ('t_mid','f8'), ('_binary_svs','i8'), 
#               ('freq','f8'), ('energy', 'f8'), ('a12','f8'), 
#               ('p', 'f8'), ('H','f8'), ('phases','f8')]

try: 
    filename=fsfile
except:
    filename = "g:100731_fs_85000_86000_10.dat.bz2"

    THIS_DATA_PATH = os.path.abspath(os.path.dirname(__file__))

    filename = "PF2_120229_MP_27233_27233_1_256.dat"
    filename = os.path.join(THIS_DATA_PATH,"PF2_120227_HMPno12_27233_27233_1.dat")
    print('file {0} was chosen: note: will use value of fsfile if it exists'.format(filename))

xx=4
shot=None
skip = 4
min_e=0.8
hold=0
time_range=None
plot=0
fsc=1.   # 1e3
sym='8' # oct  'H'  # hexagon
colorbar=None

# this is a way to check existence of variable

try:
    oldfilename
except:
    oldfilename = ""

#execfile('process_cmd_line_args.py')
import pyfusion.utils
import gzip
exec(pyfusion.utils.process_cmd_line_args())

if verbose>1: print(filename, oldfilename)
if oldfilename==filename:
    print('re-using old data - put oldfilename=None to re-read')
else:
    # this .open works for both gzip and ordinary files
    alltext = gzip.open(filename,'r').readlines()
    if len(alltext) > 1e6: print("{l} lines read ".format(l=len(alltext)))
    first_words = array([ln.split(' ')[0] for ln in alltext])
#    first_words = np.loadtxt(filename, dtype=str,usecols=[0])
    shotline = np.where(first_words == 'Shot')[0]
    if len(shotline) == 1: 
        skip = shotline[0]+1
    else: 
        print('"Shot..." line not found in {0}, default to skipping {1}'
              .format(filename, skip))

    f='f8'
    ph_dtype = [('p12',f),('p23',f),('p34',f),('p45',f),('p56',f)]
    #ph_dtype = [('p12',f)]

    # it seems as if skiprows requires a seek facility, which .gz won't allow
    print('\nLoading data from line {0} of {1}'.format(skip, filename))
    try:  # fudge the + 6 error
        ds = np.loadtxt(fname=filename, skiprows = skip, 
                        dtype= [ ('shot','i8'), ('t_mid','f8'), 
                                 ('_binary_svs','i8'), 
                                 ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                                 ('p', 'f8'), ('H','f8'), ('phases',ph_dtype)])
        oldfilename=filename
    except:
        print('problem loading - assuming + 6 error')
        print('\nLoading data from line {0} of {1}'.format(skip, filename))
        skip += 6
        ds = np.loadtxt(fname=filename, skiprows = skip,
                        dtype= [ ('shot','i8'), ('t_mid','f8'), 
                                 ('_binary_svs','i8'), 
                                 ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                                 ('p', 'f8'), ('H','f8'), ('phases',ph_dtype)])
        oldfilename=filename

# default to scatter plotting the first shot
if (shot is None): shot = ds['shot'][0]
sind=(ds['shot'] == shot).nonzero()[0]
if len(sind)==0: raise LookupError,' no data for shot %d' % shot

if min_e>0:
    subind=(ds['p'][sind] > min_e).nonzero()[0]
    if len(subind)==0: raise LookupError,' no fs meeting min_e > %.3f' % min_e
    else: sind = sind[subind]

if time_range != None:  # doesn't work!
    subind=((ds['t_mid'][sind] > time_range[0])
          & (ds['t_mid'][sind] < time_range[1])).nonzero()[0]
    if len(subind)==0: raise LookupError,' no fs meeting time_range = %.4g:.%4g' % tuple(time_range)
    else: sind = sind[subind]

if hold==0:  pl.clf()  # gets rid of extra colorbar
if plot==1: pl.scatter(ds['t_mid'][sind], fsc*freq_scale*ds['freq'][sind],\
               dot_size*np.log(ds['amp'][sind]/amp_scale),ds['a12'][sind],sym,hold=hold)
if plot and colorbar: pl.colorbar()
conditions = ", min E = {0:.3f}, amp_scale={1:.3g}".format(min_e, amp_scale)
pl.xlabel('t_mid, size is amplitude, colour is a12'+conditions)
pl.ylabel('freq')
pl.title("%s %d, %d fs" % (filename,shot,len(sind)))
if not(pl.isinteractive()): pl.show()
