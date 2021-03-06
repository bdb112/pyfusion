import numpy as np
import pylab as pl
import pyfusion
from time import time as seconds
import traceback
from time import sleep
import re
import sys

def find_data(file, target, skip = 0, recursion_level=0, debug=0):
    """ find the last line number which contains the string (regexp) target 
    work around a "feature" of loadtxt, which ignores empty lines while reading
    data,    but counts them while skipping.

    2016 - this bug was not fixed up until Feb2016, failed if there were 
    blank lines before the Shot line - fixed by double loop (see final_tries)
    
    Really should consider rewriting without loadtxt - this could also allow
    us to avoid a lot of the b'  ' stuff

    Using loadtxt at least for now:
    This way, everything goes through one reader (instead of gzip.open), simplifying,
    and allowing caching.
    The recursive part could be handled more nicely.
    about 0.14 secs to read 3500 lines, 0.4 if after 11000 lines of text
    """
    # unless dtype=str, we get a bytestring quoted as a string  "b'asd'"
    # solution is dtype='S' or stype=bytes - then -> b'asd'
    # open on the other hand lets you choose   'rb' or 'rt' (default)
    debug = max(debug, pyfusion.DEBUG)
    if debug>3: print('before loadtxt, skip={sk}'.format(sk=skip))
    lines = np.loadtxt(file,dtype=bytes,delimiter='FOOBARWOOBAR',skiprows=skip, ndmin=1)
    if debug>3: print('after, got {n} lines'.format(n=len(lines)))
    # re.match /loadtxt are confused by a python3  "b" at the beginning
    # find all the lines with the Shot target string
    wh_shotlines = np.where(
        np.array([re.match(target, line) is not None for line in lines]))[0]

    if len(wh_shotlines) == 0: 
        if (recursion_level == 0):
            raise LookupError('target {t} not found in file "{f}". Line 0 follows\n{l}'.
                              format(t=target, f=file, l=lines[0]))
        else: return(None)

    if debug>0: print('in find, recurs={r}, skip={sk}, last shotline num={las}, \nline={ln}'
                      .format(r=recursion_level, sk=skip, las=wh_shotlines[-1],ln=lines[wh_shotlines[-1]]))
    sk = wh_shotlines[-1]
    # if we are recursing, we must be checking that it is still there after skipping lines
    # return the number of the last one
    if recursion_level>0: return(sk)

    for tries in range(20):
        if debug>0: print('before find', sk, wh_shotlines[-1])

        sk1 = find_data(file, target=target, skip=sk, 
                        recursion_level=recursion_level+1)
        if debug>0: print('after find', sk, sk1)
        
        # if the target seems to be at line zero, we just about have the right answer
        #  but the first line after the skip might still be blank
        if sk1 == 0:  # so keeping looking until the target is not there
            for final_tries in range(10):
                sk += 1
                sk2 = find_data(file, target=target, skip=sk, 
                            recursion_level=recursion_level+1)
                if sk2 is None:
                    return(sk-1)
                    
            raise LookupError('too many blank lines in second search? '
                      'target {t} not found in file {f}'.
                      format(t=target, f=file))
        else:
            sk += sk1
    raise LookupError('too many blank lines? '
                      'target {t} not found in file {f}'.
                      format(t=target, f=file))

def plot_fs_DA(ds, ax=None, hold=1, ms=100, inds=None, shot=None, alpha_b=None, clim=[None, None], **sckw):
    """ ms is marker scale
        clim is the range of colours - if None and 
           there is already a collection - use its clims
           if no collections then auto scale clim
        alpha_b is the border alpha - if not None, repeat plot with Borders only at this alpha
           (usually 1)

    colls=[ch for ch in plt.gca().get_children() if hasattr(ch,'get_clim')]

    """
    ax = ax if ax is not None else pl.gca()
    fig = ax.get_figure()
    if hold == 0:
        fig.clf() # clear the colorbar()
        ax = pl.gca()
    do_cbar =True
    if clim[0] is None:
        colls=[ch for ch in ax.get_children() if hasattr(ch,'get_clim')]
        if len(colls) > 0:
            clim = colls[0].get_clim()
            do_cbar = False
    inds = np.arange(len(ds['shot'])) if inds is None else inds
    if shot is not None:
        inds = np.where(ds['shot'] == shot)[0] 
    ord = np.argsort(ds['a12'][inds])  # sort so that the highest a12 lies on top
    inds = inds[ord]
    scp = ax.scatter(ds['t_mid'][inds],ds['freq'][inds],ms*ds['a12'][inds], ds['amp'][inds],
                         vmin=clim[0], vmax=clim[1], **sckw)
    if alpha_b is not None:
        #sckw.update(dict(alpha=alpha_b, facecolor="r"))
        scp.set_facecolor("none")
        scp.set_alpha(alpha_b)
        #ax.scatter(ds['t_mid'][inds],ds['freq'][inds],ms*ds['a12'][inds], ds['amp'][inds],
        #           vmin=clim[0], vmax=clim[1], **sckw)
                   
        
    shotstr = str('{sfr}..{sto}'.format(sfr=ds['shot'][0], sto=ds['shot'][-1]))
    if len(np.unique(ds['shot'])) == 1:
        shotstr = str(ds['shot'][0])
    if shot is not None:
        shotstr = str(shot)
    if hasattr(ds, 'name'):
        shotstr = shotstr + '[{name}]'.format(name=ds.name)
    elif hasattr(ds, 'keys') and isinstance(ds['info']['comment'][0], str):
        shotstr = shotstr + ' [{name}]'.format(name=ds['info']['comment'][0])

    fig.suptitle('{shotstr}\n colour is amp, size is a12'.format(shotstr=shotstr))
    if do_cbar:
        fig.colorbar(scp)


def read_text_pyfusion(files, target=b'^Shot .*', ph_dtype=None, plot=pl.isinteractive(), ms=100, hold=0, debug=0, quiet=1,  maxcpu=1, exception = Exception):
    """ Accepts a file or a list of files, returns a list of structured arrays
    See merge ds_list to merge and convert types (float -> pyfusion.prec_med
    """
    regulator = pyfusion.utils.Regulator(maxcpu)
    st = seconds(); last_update=seconds()
    file_list = files
    if len(np.shape(files)) == 0: file_list = [file_list]
    f='f8'
    if ph_dtype is None: ph_dtype = [('p12',f),('p23',f),('p34',f),('p45',f),('p56',f)]
    #ph_dtype = [('p12',f)]
    ds_list =[]
    comment_list =[]
    count = 0
    for (i,filename) in enumerate(file_list):
        regulator.wait()
        if seconds() - last_update > 10:
            last_update = seconds()
            tot = len(file_list)
            print('read {n}/{t}: ETA {m:.1f}m {f}'
                  .format(f=filename, n=i, t=tot,
                          m=(seconds()-st)*(tot-i)/float(60*i)))

        try:
            if (isinstance(target,str) or isinstance(target,bytes)): 
                skip = 1+find_data(filename, target, debug=debug)
            elif isinstance(target, int): 
                skip = target
            else:
                raise Exception('target ({target}) is not recognised'.format(target=target))
            if quiet == 0:
                print('{t:.1f} sec, loading data from line {s} of {f}'
                      .format(t = seconds()-st, s=skip, f=filename))
            #  this little bit to determine layout of data
            # very inefficient to read twice, but in a hurry!
            if debug>2: print('skiprows = \n', skip-1)
            txt = np.loadtxt(fname=filename, skiprows=skip-1, dtype=bytes, 
                             delimiter='FOOBARWOOBAR',ndmin=1)
            header_toks = txt[0].split()
            # look for a version number first
            if header_toks[-1][-1] in b'0123456789.':
                version = float(header_toks.pop())
                if b'ersion' not in header_toks.pop():
                    raise ValueError('Error reading header in {f}'
                                     .format(f=filename))
            else: version=-1  # pre Aug 12 2013
            # noticed that the offset moved in 2015 - when did it  happen?
            phase_offs = -4 if sys.version>'3,' else -2
            # is the first character of the 2nd last a digit?
            if header_toks[phase_offs][0] in b'0123456789': 
                if pyfusion.VERBOSE > 0: 
                    print('header toks', header_toks)
                    print('found new header including number of phases')
                n_phases = int(header_toks[phase_offs])
                ph_dtype = [('p{n}{np1}'.format(n=n,np1=n+1), f) for n in range(n_phases)]
                
            if 'frlow' in header_toks:  # add the two extra fields
                fs_dtype= [ ('shot','i8'), ('t_mid','f8'), 
                            ('_binary_svs','u8'),    # f16 - really want u8 here,  but npyio 
                                                      #has problem converting 10000000000000000000000000
                                                      #OverflowError: Python int too large to convert to C long
                                                      # doesn't happen if text is read in directly with loadtxt
                            ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                            ('p', 'f8'), ('H','f8'), 
                            ('frlow','f8'), ('frhigh', 'f8'),('phases',ph_dtype)]
            else:
                fs_dtype= [ ('shot','i8'), ('t_mid','f8'), 
                            ('_binary_svs','u8'), 
                            ('freq','f8'), ('amp', 'f8'), ('a12','f8'),
                            ('p', 'f8'), ('H','f8'), ('phases',ph_dtype)]

            if version > 0.69:  # don't rely on precision
                fs_dtype.insert(-1,('cpkf', 'f8'))  # -1 is 1 before the end
                fs_dtype.insert(-1,('fpkf', 'f8'))  # they appear in this order
                
            if pyfusion.VERBOSE > 0: 
                print(version, fs_dtype, '\n')

            ds = np.loadtxt(fname=filename, skiprows = skip, 
                            dtype= fs_dtype, ndmin=1)  # ENSURE a 1D array

            if len(ds) > 0:
                ds_list.append(ds)
                count += 1
                # npz reads in python 2 can't cope with unicode - don't report errors unless really debugging
                comment_list.append(filename.encode(errors=['ignore','strict'][pyfusion.DBG() > 5]))
            else:
                print('no data in {f}'.format(f=filename))

        except ValueError as reason:
            print('Conversion error while processing {f} with loadtxt - {reason} {args}'
                  .format(f=filename, reason=reason, args=reason.args))
            traceback.print_exc()

        except exception as info:
            print('Other exception while reading {f} with loadtxt - {info} {a}'.format(f=filename, info=info, a=info.args))
            traceback.print_exc()
    print("{c} out of {t} files".format(c=count, t=len(file_list)))
    if plot>0 and len(ds_list)>0: 
        plot_fs_DA(ds_list[0], ms=ms)
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
                newtype = np.dtype('uint64')

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

    if not 'phorig' in dd:  # preserve the original phase ch0
        # as an integer8 scaled by 10 to save space
        # this way, we can play with phases (invert etc) but
        # can always check to see what has been changed
        # at list the first element.
        dd['phorig'] = np.array(dd['phases'][:,0]*10).astype('int8')

    # put the comments in a dictionary, so that operations on arrays won't be atttempted
    dd['info'] = {'comment': np.array(comment_list)}
    print("############# comment_list", comment_list)

    return(dd)
    
if __name__ == "__main__":
    from glob import glob
    """    import doctest
    doctest.testmod()
    """
    (ds_list, comment_list) = read_text_pyfusion(glob('pyfusion/test_files/bad_PF2_dat/PF*2'))
    print('expected result is 3 exceptions, and 2 out of 5 files')
    if len(ds_list)!=2: print('test failed')
