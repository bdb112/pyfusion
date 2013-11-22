import numpy as np
import pylab as pl
import pyfusion
from time import time as seconds
import re
import sys

def find_data(file, target, skip = 0, recursion_level=0, debug=0):
    """ find the last line number which contains the string (regexp) target 
    work around a "feature" of loadtxt, which ignores empty lines while reading data
    but counts them while skipping.
    This way, everything goes through one reader (instead of gzip.open), simplifying,
    and allowing caching.
    The recursive part could be handled more nicely.
    about 0.14 secs to read 3500 lines, 0.4 if after 11000 lines of text
    """
    lines = np.loadtxt(file,dtype=str,delimiter='FOOBARWOOBAR',skiprows=skip)
    wh_shotlines = np.where(
        np.array([re.match(target, line)!=None for line in lines]))[0]
    if debug>0: print('main, rec, skip=', 
                      recursion_level, skip, wh_shotlines[-1],lines[wh_shotlines[-1]])
    if len(wh_shotlines) == 0: 
        raise LookupError('target {t} not found in file "{f}". Line 0 follows\n{l}'.
                          format(t=target, f=file, l=lines[0]))
    sk = wh_shotlines[-1]
    if recursion_level>0: return(sk)

    for tries in range(10):
        if debug>0: print(sk, wh_shotlines[-1], lines[sk])

        sk1 = find_data(file, target=target, skip=sk, 
                        recursion_level=recursion_level+1)
        if debug>0: print(sk, sk1)
        
        if sk1 ==0: return(sk)
        else: sk += sk1

    raise LookupError('too many blank lines?'
                      'target {t} not found in file {f}'.
                      format(t=target, f=file))


def read_text_pyfusion(files, target='^Shot .*', ph_dtype=None, plot=pl.isinteractive(), ms=100, hold=0, debug=0, quiet=1,  exception = Exception):
    """ Accepts a file or a list of files, returns a list of structured arrays
    See merge ds_list to merge and convert types (float -> pyfusion.prec_med
    """
    st = seconds(); last_update=seconds()
    file_list = files
    if len(np.shape(files)) == 0: file_list = [file_list]
    f='f8'
    if ph_dtype == None: ph_dtype = [('p12',f),('p23',f),('p34',f),('p45',f),('p56',f)]
    #ph_dtype = [('p12',f)]
    ds_list =[]
    comment_list =[]
    count = 0
    for (i,filename) in enumerate(file_list):
        if seconds() - last_update > 30:
            last_update = seconds()
            tot = len(file_list)
            print('reading {n}/{t}: ETA {m:.1f}m {f}'
                  .format(f=filename, n=i, t=tot,
                          m=(seconds()-st)*(tot-i)/float(60*i)))

        try:
            if pl.is_string_like(target): 
                skip = 1+find_data(filename, target,debug=debug)
            else: 
                skip = target
            if quiet == 0:
                print('{t:.1f} sec, loading data from line {s} of {f}'
                      .format(t = seconds()-st, s=skip, f=filename))
            #  this little bit to determine layout of data
            # very inefficient to read twice, but in a hurry!
            txt = np.loadtxt(fname=filename, skiprows=skip-1, dtype=str, 
                             delimiter='FOOBARWOOBAR')
            header_toks = txt[0].split()
            # look for a version number first
            if header_toks[-1][-1] in '0123456789.':
                version = float(header_toks.pop())
                if 'ersion' not in header_toks.pop():
                    raise ValueError('Error reading header in {f}'
                                     .format(f=filename))
            else: version=-1  # pre Aug 12 2013

            # is the first character of the 2nd last a digit?
            if header_toks[-2][0] in '0123456789': 
                if pyfusion.VERBOSE > 0: 
                    print('found new header including number of phases')
                n_phases = int(header_toks[-2])
                ph_dtype = [('p{n}{np1}'.format(n=n,np1=n+1), f) for n in range(n_phases)]
                
            if 'frlow' in header_toks:  # add the two extra fields
                fs_dtype= [ ('shot','i8'), ('t_mid','f8'), 
                            ('_binary_svs','f8'),    # really want i8 here, but npyio 
                                                      #has problem converting 10000000000000000000000000
                                                      #OverflowError: Python int too large to convert to C long
                                                      # doesn't happen if text is read in directly with loadtxt
                            ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                            ('p', 'f8'), ('H','f8'), 
                            ('frlow','f8'), ('frhigh', 'f8'),('phases',ph_dtype)]
            else:
                fs_dtype= [ ('shot','i8'), ('t_mid','f8'), 
                            ('_binary_svs','i8'), 
                            ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                            ('p', 'f8'), ('H','f8'), ('phases',ph_dtype)]

            if version > 0.69:  # don't rely on precision
                fs_dtype.insert(-1,('cpkf', 'f8'))  # -1 is 1 before the end
                fs_dtype.insert(-1,('fpkf', 'f8'))  # they appear in this order
                
            if pyfusion.VERBOSE > 0: 
                print(version, fs_dtype)

            ds_list.append(
                np.loadtxt(fname=filename, skiprows = skip, 
                           dtype= fs_dtype, ndmin=1)  # ENSURE a 1D array
            )
            count += 1
            comment_list.append(filename)
        except ValueError, info:
            print('Conversion error while reading {f} with loadtxt - {info}'.format(f=filename, info=info))

        except LookupError, info:
            print('Lookup error while reading {f} with loadtxt - {info}'.format(f=filename, info=info))

    print("{c} out of {t} files".format(c=count, t=len(file_list)))
    if plot>0: 
        ds = ds_list[0]
        if hold == 0: pl.clf() # for the colorbar()
        pl.scatter(ds['t_mid'],ds['freq'],ms*ds['a12'],ds['amp'])
        pl.title('{s}, colour is amp, size is a12'.format(s=ds['shot'][0]))
        pl.colorbar()
    return(ds_list, comment_list)

def merge_ds(ds_list, comment_list=[], old_dd=None, debug=True, force=False):
    """ Take a list of structured arrays, and merge into one
    Adding to an existing dd is not fully tested - may be memory intensive
    and does not check for keys being different in the ds and dd
    """
    if len(np.shape(ds_list)) == 0: 
        raise ValueError("{d} should be a list".format(d=ds_list))

    if type(ds_list[0]) == type({}): keys = np.sort(ds_list[0].keys())
    elif type(ds_list[0]) == np.ndarray: keys = np.sort(ds_list[0].dtype.names)

    if old_dd is None: 
        dd = {}
    else: 
        if debug>0: print('appending')
        dd = old_dd
        ddkeys = np.sort(dd.keys())
        if (len(ddkeys) != len(keys)) or (not np.char.equal(keys, ddkeys)):
            msg = str('keys are not the same: \n {kds}\n{kdd}'
                      .format(kds = keys, kdd = ddkeys))
            if force: pyfusion.utils.warn(msg)
            else: raise LookupError(msg)

    #  for each key in turn, make an array from the ds_list[0], then
    #  extend it with ds_list[1:]
    for k in keys:
        # get rid of the structure/record stuff, and convert precision
        # warning - beware of data with very high dynamic range!
        if old_dd is None:
            if np.issubdtype(type(ds_list[0][k][0]), int): 
                newtype = np.dtype('int32')
            elif np.issubdtype(type(np.array(ds_list[0][k].tolist()).flatten()[0]), float): 
                newtype = pyfusion.prec_med
            else: 
                print("defaulting {0} to its type in ds_list".format(k))
                newtype = type(ds_list[0][k][0])

            # make sure t_mid is at least f32 (so that 100 sec shot records
            # accurately to a few usec
            if k == 't_mid' and np.issubdtype(newtype, np.dtype('float32')):
                newtype = np.dtype('float32')
            # until binary svs are properly binary, need 64 bits for 10 channels or more
            if k == '_binary_svs' and np.issubdtype(newtype, np.dtype(int)):
                newtype = np.dtype('int64')

            arr = np.array(ds_list[0][k].tolist(),dtype=newtype)# this gets rid 
        # of the record stuff and saves space (pyfusion.med_prec is in .cfg)
            start = 1 # the first DS is now in the array
        else: 
            arr = dd[k].copy()  # could be wasteful?
            start = 0 # the first DS is NOT in the array - it is the old_dd 


        for (ind, ds) in enumerate(ds_list[start:]):
            if debug>0: 
                print(ind, k),
                if k == 'shot': sys.stdout.flush()
            oldlen = len(arr)
            if len(np.shape(arr))==1:  # 1D data
                arr.resize(oldlen+len(ds[k]))
                arr[oldlen:] = ds[k][:]
            else:
                arr.resize(oldlen+len(ds[k]),len(arr[0])) # 2D only?
                """ 13 secs to  merge two lots of 500k lines (5 phases)
                for j in range(len(arr[0])):
                    arr[oldlen:,j] = np.array(ds[k].tolist())[:,j]
                """   
                # this version is 3.1 secs/500k lines
                float_phases = np.array(ds[k].tolist())
                for j in range(len(arr[0])):
                    arr[oldlen:,j] = float_phases[:,j]

        dd.update({k: arr})

    if not dd.has_key('phorig'):  # preserve the original phase ch0
        # as an integer8 scaled by 10 to save space
        # this way, we can play with phases (invert etc) but
        # can always check to see what has been changed
        # at list the first element.
        dd['phorig'] = np.array(dd['phases'][:,0]*10).astype('int8')

    # put the comments in a dictionary, so that operations on arrays won't be atttempted
    dd['info'] = {'comment': np.array(comment_list)}


    return(dd)
    
