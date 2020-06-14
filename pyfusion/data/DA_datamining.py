from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from time import time as seconds
import os
import sys
from warnings import warn
import six  # just for is_string_like fix

"""
Note: In np.load() allow_pickle defaults to False in 1.17 c.f. True in 1.16.1
    This is only relevant to saving objects - not needed for simple arrays of 
     primitive types.  The only such type in pyfusion data is the params dic

Note: in programming, be careful not to refer to .da[k] unnecessarily
if it is not loaded - typically, if you plan to test it then use it,
save to a var first, then test, then use it.  see "dak" below
"""

# Want to avoid depending on pyfusion, but use it if it is there
# First, arrange a debug one way or another
try: 
    from pyfusion.debug_ import debug_
except:
    def debug_(debug, msg='', *args, **kwargs):
        if debug > 0:
            print('attempt to debug {msg}'.format(msg=msg) +
                  " need boyd's debug_.py to debug properly")


def myload(filename, allow_pickle=True):
    """ allow_pickle defaults to false in 1.16.3  -  CVE-2019-6446
    need to allow for old versions that don't have the allow_pickle keyword 
    In future, save the params as a json string instead to totally avoid this.
    """

    from numpy import __version__
    kwargs = dict(allow_pickle=allow_pickle) if __version__ > '1.16.1' else {}
    return(np.load(filename, **kwargs))

class Masked_DA():
    """  A virtual sub dictionary of a DA, contained in the 'masked' attribute
         and returning applicable (valid_keys) elements, masked by DA.da['mask']
         to have Nans in the positions where mask = False or 0

    An important side effect is to add the mask array to the main dictionary
    Probably should NOT be a subclass - we don't want to do unnecessary copying.
    
Parameters:
  valid_keys: keys to which mask should be applied.
  mask: An array (usualy 2D) of the same shape as the data, 
    is usually set at a later stage, when the quality or error criteria
    are evaluated.
  baseDA: not sure why this is required, because this object is usually
    attached to an existing DA - needs some thought.

Example:
  >>> from pyfusion.data.DA_datamining import Masked_DA, DA
  >>> mydDA=DA('20160310_9_L57',load=1)  # needs to be loaded for this operation
  >>> da.masked=Masked_DA(valid_keys=['Te','I0','Vp'], baseDA=myDA)
  >>> myDA.da['mask']=-myDA['resid']/abs(myDA['I0'])<.35
  >>> clf();plot(myDA.masked['Te']);ylim(0,100)

    """
    def __init__(self, valid_keys=[], baseDA=None, mask=None):
        if baseDA is None:
            #raise ValueError('Need an existing DA or dict as DA arg')
            print('Need an existing DA or dict as DA arg - should fix!~~')
        # nothing much to do
        self.DA = baseDA  # this doesn't copy - just a convenience 
        if mask is not None:
            if len(np.shape(mask)) == 0:
                mask = mask + np.zeros(shape=np.shape(self.DA.da[valid_keys[0]]))
            self.DA.da['mask'] = mask
        self.valid_keys = valid_keys

    def keys(self):
        """  Return the keys for elements with validity masking only.
        To see all elements, including raw data, use keys() on the parent DA.
        object."""
        if 'mask' not in self.DA :
            print('no mask set')
            return([])
        else:
            return(self.valid_keys)

    def __getitem__(self, key):
        if 'mask' not in self.DA:
            raise KeyError(key, 'no mask set')
        elif key not in self.valid_keys:
            raise KeyError(key)
        elif np.shape(self.DA['mask']) != np.shape(self.DA[key]):
            ms = np.shape(self.DA['mask'])
            ds = np.shape(self.DA[key])
            raise ValueError('mask shape {ms} does not match data shape {ds}'.format(ms=ms, ds=ds))
        else:
            tmp = self.DA.da[key].copy()
            tmp = np.where(self.DA['mask'] == 1, tmp, np.nan)
            return(tmp)
            
def info_to_bytes(inf):
    """ Eventually this should deal with bytes/unicode compatibility"""
    #for k in inf:  # just do comment for now
    # can't do isinstance(xx, unicode) as unicode is not defined in P3
    if sys.version > '3,':
        for c in inf['comment']:
            if not isinstance(c, bytes):
                print("warning - unicode in info, can't read in python2, even 2.7")
                # do nothing for now
    return(inf)
        
def mylen(ob):
    """ return the length of an array or dictionary for diagnostic info """
    if type(ob) == type({}):
        return(len(ob.keys()))
    elif ob is None:
        return(0)
    else:
        try:
            return(len(ob))
        except:
            return(-999)
# try to use psutils to watch memory usage - be quiet if it fails
try:
    import psutil

    def report_mem(prev_values=None, msg=None, verbose=0):
        """ Show status of phy/virt mem, and if given previous values, 
        report differences
        requires the psutil module - avail in apt-get, but I used pip
        """
        if msg is None: msg=''
        else: msg += ': '
        if type(prev_values) == type(''):  # catch error in prev_values
            msg += prev_values
            prev_values = None
            print('need msg= in call to report_mem')

        # available in linux is free + buffers + cached
        if hasattr(psutil,'swap_memory'):  # new version (0.7+)
            pm = psutil.virtual_memory().available     # was psutil.avail_phymem()
            vm =  psutil.swap_memory().free            # was psutil.avail_virtmem()
        else:
            pm = psutil.phymem_usage().free # was .avail_phymem()
            vm =psutil.virtmem_usage().free # avail_virtmem()

        tim = seconds()
        if verbose > -1:
            print('{msg}{pm:.3g} GB phys mem, {vm:.3g} GB virt mem avail'
                  .format(msg=msg, pm=pm/1e9, vm=vm/1e9)),

        if prev_values is None:
            if verbose > -1:
                print()
        else:
            if verbose > -1:
                print('- dt={dt:.2g}s, used {pm:.2f} GB phys, {vm:.2g} GB virt'
                      .format(pm=(prev_values[0] - pm)/1e9,
                              vm=(prev_values[1] - vm)/1e9, dt = tim-prev_values[2]))
        return((pm,vm,tim))
except ImportError:
    print('need psutil to get useful info about memory usage')
    def report_mem(prev_values=None, msg=None):
        return((None, None))


def process_file_name(filename):
    """ Allow shell shortcuts such as ~/ and env var expansion 
-   Note: not tested in windows """
    if '$DAPATH' in os.path.split(filename):
        try:
            import pyfusion
            filename = filename.replace('$DAPATH',pyfusion.config.get('global','DAPATH'))
        except importError:
            print('Warning - apparently running outside of pyfusion - will check environment')

    # expand env and home
    fname = os.path.expanduser(os.path.expandvars(filename))
    if not (os.path.exists(fname)) and ('.npz' not in fname):
            fname += '.npz'
    return(fname)


def append_to_DA_file(filename, new_dict, force=False):
    """ 
    Adds a new variable to the file - more like 'update' than append

    Opens filename with mode=a, after checking if the indx variables align
    force=1 ignores checks for consistent length c.f. the var shot.

    Works with a DA file, in contrast to DA.append() which extends a DA

Parameters:
    filename: file to append to
    new_dict: dictionary with new data to append
    force: try to continue if there is a mismatch error
Returns:
    no return - side effect is to add a new variable to a DA file
Raises:
    ValueError: if new arrays don't match old.

Example:
    >>> append_to_DA_file('DAX.npz',dict(N=dd['N'])     # simple
    >>> append_to_DA_file('foo.npz',dict((k, mydict[k]) for k in ['N','M']))

    Not a member of the class DA, because the class has memory copies of the
    file, so it would be confusing.
    """
    import zipfile, os

    # The file is loaded into dd, and the new dictionary is new_dict
    # Add .npz if the name given does not exist, and it is not already
    dd = myload(process_file_name(filename))
    error = None
    if 'indx' in dd.keys() and 'indx' in new_dict.keys():
        check = dd['indx']
        if (new_dict['indx']!=check).all():
            error=('mismatched indx vars in ' + filename)

    else:
        print('indx missing from one or both dictionaries - just checking size')
        check=dd['shot']
        if (len(check) != len(new_dict[list(new_dict)[0]])):
            error=('mismatched var lengths in ' + filename)

    if error is not None:
        if force:  warn(error)
        else:      raise ValueError(error)

    zf = zipfile.ZipFile(filename,mode='a')
    try:
        import pyfusion
        tmp =  pyfusion.config.get('global', 'my_tmp')
    except ImportError:
        print('outside pufusion, so assume /tmp')
        tmp = '/tmp'
    for key in new_dict.keys():
        if key in zf.namelist():
            print('member {k} already exists - will supersede but not remove it'
                  .format(k=key))
        tfile = tmp+'/'+ key+'.npy'
        np.save(tfile, new_dict[key])
        zf.write(tfile,arcname=key,compress_type=zipfile.ZIP_DEFLATED)
        os.remove(tfile)
    zf.close()

class DA():
    """ 
Class to handle and save data in a special dictionary of arrays 
referred to hereafter as a "DA".
 * Can deal with databases larger than memory, by using load=0
 * Faster to use if load=1, but if you subselect by using extract
   you get the speed for large data sets (once extract is done).
 * Extract can be used over and over to get different data set selections.

Args:
  fileordict: An .npz file containing a DA object or a dictionary of arrays 
    sharing a common first dimension, including the result of a 
    loadtxt(dtype=...) command. The filename is processed for env vars ~/ etc,
    but sometimes this seems to substitute the path of the DA module? (bug)
  load: 1 will immediately load into memory, 0 will defer load allowing 
    some operations (but slowly) without consuming memory.
  mainkey: The main key, not necessarily a unique identifier - e.g it 
    can be shot.
  limit: Decimates the data when loaded into memory (via load=1). It is 
    the most effective space saver, but you need to reload if more (or a 
    different subselection of data) is needed.  The alternative is to 
    downselect by using 'extract=' (but this applies only to the variables 
    extracted into namespace (e.g. locals())

Returns:
    A DA object as described above

Raises:
    KeyError, ValueError, LookupError:

*Experimental* new feature allows use of the DA object itself as a 
  dictionary (e.g. DA59['shot']).
For more info type help(DA)

Note: This is my prototype of google style python sphinx docstrings - based on 
    http://www.sphinx-doc.org/en/stable/ext/example_google.html
    Had to include 'sphinx.ext.napoleon' in documentation.conf.py to get the
    parameters on separate lines.
    """
    def __init__(self, fileordict, debug=0, verbose=0, load=0, limit=None, mainkey=None):
        # may want to make into arrays here...
        self.debug = debug
        self.verbose = verbose
        self.loaded = False
        
        start_mem = report_mem(msg='init', verbose=self.verbose)
        # not sure if it is a good idea to accept a DA - don't allow for now.
        # why would you want to make a DA from a DA?
        if isinstance(fileordict, dict) or hasattr(fileordict,'zip'): #  or (hasattr(fileordict, 'da')):  (isinstance(fileordict, dict) or hasattr(fileordict,'zip')
            # look for illegal chars in keys (ideally should be legal names)
            baddies = '*-()[]'   # e.g. shaun uses sqrt(ne)
            bads = [ch in '_'.join(fileordict.keys())for ch in baddies]
            if True in bads:
                fileordict = fileordict.copy()
                for k in fileordict.keys():
                    newk = k.translate(None,baddies) # with a table of none, just delete baddies
                    if newk != k:
                        print('*** renaming {k} to {newk}'.format(k=k, newk=newk))
                        fileordict.update({newk:fileordict.pop(k)})
                        debug_(debug, 1, key='rename')

            self.da = fileordict
            self.loaded = True
            self.name = 'dict'
        else:
            self.name = process_file_name(fileordict)
            self.da = myload(self.name)
            self.loaded = False  # i.e. not really loaded yet - just have the zipfile table

        self.keys = self.da.keys  # self.keys used to be a list, now a function
        if 'dd' in self.da:  # old style, all in one - is this like a loadtxt return?
            print('old "dd" object style file')  # should have a test example
            self.da = self.da['dd'].tolist()
            self.keys = self.da.keys

        # make the info available to self.
        # needs different actions if npzfile and not yet loaded
        # we don't load yet, as we may want to decimate
        if 'info' in self.da:
            if (hasattr(self.da['info'],'dtype') and                 self.da['info'].dtype == np.dtype('object')):
                self.infodict = self.da['info'].tolist()
            else:
                self.infodict = self.da['info']
        else:
            self.infodict = {}

        try:  # silent attempt to get host info - avoids pyfusion dependence
            import pyfusion
            self.infodict['host'] = pyfusion.utils.host()
        except:
            pass
        self.__doc__ += 'foo\n'  # this isn't accessible to the ? or help function

        self.mainkey = mainkey  # may be None!
        debug_(self.debug, 3)
        if self.mainkey is None:
            if 'mainkey' in self.infodict.keys():
                self.mainkey = self.infodict['mainkey']
            else:
                if 'params' in self.keys():
                    raise ValueError('This is probably a pyfusion cached data file')
                elif 'shot' in self.da:
                    self.mainkey = 'shot'
                else:
                    self.mainkey = self.da.keys()[0]

        self.len = len(self.da[self.mainkey])

        if (limit is None) or (self.len < abs(limit)):
            self.sel = None
        else:  # decimate to a maximum of approx "limit"
            #if load != 0:
            #    raise ValueError('decimate not applicable to ' + self.name)
            if limit<0: 
                print('repeatably' ),
                np.random.seed(0)  # if positive, should be random
                                   # negative will repeat the sequence
            else: print('randomly'),

            print('decimating from sample of {n}'.format(n=self.len))
            self.sel = np.where(np.random.random(self.len)
                                < float(abs(limit))/self.len)[0]

        if 'mask' in self.da:  # need to load first, regardless
            load = 1
            if self.verbose > -1:
                print('Autoloading as data has a mask enabled')

        if load == 0:  
            self.loaded = False
        else:  # load must be True
            if self.loaded == 0:
                self.loaded = self.load()

        if 'mask' in self.da:  # give it the Masked_DA property
            valid_keys = self.infodict.get('valid_keys',[])
            self.masked = Masked_DA(valid_keys, self)

        # add an index if there is not one already.  If there is one, 
        # warn if it is not sequential from 0 to len of mainkey entry
        # useful for doing multiple search criteria - 
        # extract data accoring to criterion 1, then select according to crit 1
        # and reextract

        if not 'indx' in self.da:
            if not self.loaded:
                self.load()
            self.da.update(dict(indx=np.arange(len(self.da[self.mainkey]))))

        else:
            indtmp = self.da['indx']
            if (len(indtmp) != len(self.da[self.mainkey]) or
                (min(indtmp) != 0) or (len(indtmp) > 1 and np.max(np.diff(indtmp))>1)):
                print("**** warning - index is not montonic from 0 ***** ")

        self.infodict.update({'mainkey':self.mainkey}) # update in case it has changed
        if type(self.da) == dict:
            self.update({'info': self.infodict}, check=False)



        start_mem = report_mem(start_mem, verbose=self.verbose)
    #shallow_copy = try:if da.copy

## emulate a dictionary for convenience. See https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types
# implement __getitem__ and __iter__ so that 'shot' in DAHJ  works.
    def __getitem__(self,k):
        if k.lower() == 't':  # the corrected time - only at top level.
            t_zero = self.da['info'].get('t_zero', None)
            if t_zero is None:
                print('*** Warning - no time correction found! - returning raw time ***')
                t_zero = 0
            return(self.da['t_mid'] -  t_zero)
        else:
            return(self.da[k])

    def __iter__(self):   # surprised I needed iter() here
        keys = list(self.da.keys())
        if 't_zero' in self.da['info'] and 't_mid' in self.da:
            keys.append('t')
        return(iter(keys))

    def update(self, new_dict, check=True):
        """ Add a new variable to the dictionary.  Better than simply updating
        dd, as it allows length check and updates the list of keys.
        """
        dlen = len(self.da[self.mainkey])
        new_keys = []
        for nkey in new_dict.keys():
            if nkey in self.da.keys():
                if self.verbose > -1: print('replacing {k}'.format(k=nkey))
            else:
                new_keys.append(nkey)

            if check and len(new_dict[nkey]) != dlen:
                raise LookupError('key {k} length {lk} does not match {l}'
                                  .format(k=nkey, lk = len(new_dict[nkey]), 
                                          l=dlen))
        self.da.update(new_dict)    


    def append(self, dd):
        """ append the data arrays in dd to the data arrays in self - i.e.
        extend the existing arrays.  Typical use is in serial processing of
        a range od shots. 
      See also append_to_DA_file to add an extra variable
        """
        # check keys to make sure they match
        for k in self.da.keys():
            if k not in dd.keys():
                raise KeyError('key {k} not in dd keys: {keys}'
                                  .format(k=k, keys=dd.keys()))
        for k in dd.keys():
            if k not in self.da.keys():
                raise KeyError('key {k} not in DA keys: {keys}'
                                  .format(k=k, keys=self.da.keys()))
        for k in self.da.keys():
            if hasattr(dd[k],'keys'):  # check if the dd entry [k] is itself a dict
                if self.verbose > -1:
                    print('Info: dd entry {k} is a dictionary with keys {ks}'
                          .format(k=k, ks=dd[k].keys()))
                for kk in self.da[k]:
                    if kk in dd.keys():  # if it is there, append
                        self.da[k][kk] = np.append(self.da[k][kk], dd[k][kk])
                    else:
                        self.da[k][kk] = dd[k][kk]  # else insert
                
            else:
                self.da[k] = np.append(self.da[k], dd[k], 0)
                
        self.len = len(self.da[self.mainkey])
        if self.verbose>0: 
            print('added {dl} instances to make a total of {tl}'
                  .format(dl=len(dd[self.mainkey]), tl = self.len))


    def write_arff(self, filename, keys=[]):
        """ keys is a list of keys to include, and empty list includes all
        """
        from .write_arff import write_arff
        write_arff(self, filename, keys)


    def to_sqlalchemy(self,db = 'sqlite:///:memory:',mytable='fs_table', n_recs=1000, newfmts={}, chunk=1000):
        from .write_arff import split_vectors

        """ Write to an sqlachemy database 
            chunk = 500: 2000 34 element recs/sec to (big) sqllite file, 
                    1600/sec to mysql.  cat file|mysql junk is ~ 16,000/sec
                    Using load data infile -> 25,000/s (for file on server!)

                    mysql> select * from fs_table into outfile 'foo1';
                    Query OK, 100000 rows affected (0.78 sec) (see format below)

                    mysql> load data INFILE "foo1" into table fs_table;
                    Query OK, 100000 rows affected (4.09 sec)


            'mysql://bdb112@localhost/junk'  (mysql needs nans cvtd to nul

            This is a proof of principle - won't work with other than 
            numeric scalars at the moment - DA.pop('phases'); DA.pop('info')
        """        
        """ Format for mysql load data infile
'\n	27153	0.1	-1	0.285	11.7753	1	10010	-0.000908817	2.75	0.074	18.3	0.0156897	-0.118291	1.25	27153	\n	-1	-1	-1	\n	0.373338	\n	-0.00219764	0.001536	\n  etc'
        """

        import sqlalchemy as SA
        def cvt(val):
            if np.isnan(val): return(None)
            elif np.issubdtype(val, np.float):
                return(float(val))
            elif np.issubdtype(val, np.int):
                return(int(val))
            else:
                if self.debug>0: print('unrecognised type {t} in cvt'
                                       .format(t=type(val)))
                return(None)

        # define the table
        self.engine = SA.create_engine(db, echo=self.debug>2)
        self.metadata = SA.MetaData()
        self.mytable = SA.Table(mytable, self.metadata)

        dd = self.copyda()
        sub_list = split_vectors(dd, newfmts=newfmts)

        (dbkeys,dbtypes)=([],[])
        for k in np.sort(list(dd.keys())):
            arr = dd[k]
            typ = None
            print(k)
            if hasattr(arr,'dtype'):
                if np.issubdtype(arr.dtype, np.int): 
                    typ = SA.Integer
                elif np.issubdtype(arr.dtype, np.float): 
                    typ = SA.Float
                else:
                    print('unknown dtype {d}'.format(d=arr.dtype))
                    
                if typ != None: # if it gets here, it is recognised
                    dbkeys.append(k)
                    dbtypes.append(typ)
                    self.metadata.tables[mytable].append_column(SA.Column(k, typ))
                    debug_(self.debug, 2)

            if self.debug>0: print(self.metadata.tables)

        if len(dbkeys)==0: return('nothing to create')
        self.metadata.create_all(self.engine)
        conn=self.engine.connect()
        if self.len > n_recs: print('Warning - only storing n_rec = {n} records'
                                    .format(n=n_recs))
        for c in range(0,min(int(n_recs),len(dd[dbkeys[0]])),chunk):
            print(c, min(c+chunk, self.len))
            # for each chunk, make a list if dicts containing the fs record
            lst = []
            for i in range(c,min(c+chunk, min(self.len,n_recs))):
                dct = {}
                for (k,key) in enumerate(dbkeys): 
                    dct.update({key: cvt(dd[key][i])})
                lst.append(dct)    
            if self.debug>0: print(lst)
            conn.execute(self.mytable.insert(),lst)
                
                

    def copyda(self, force = False):
        """ make a deepcopy of self.da
        typically dd=DAxx.copy()
        instead of dd=DAxx.da - which will make dd and DAxx.da the same thing (not usually desirable)
        """
        from copy import deepcopy
        if (not force) and (self.len > 1e7): 
            quest = str('{n:,} elements - do you really want to copy? Y/^C to stop'
                        .format(n=self.len))
            if plt.isinteractive():
                if 'Y' not in raw_input(quest).upper():
                    1/0
            else: 
                print(quest)

        if self.loaded == 0: self.load()
        start_mem = report_mem(msg='copying', verbose=self.verbose)
        cpy = deepcopy(self.da)
        report_mem(start_mem, verbose=self.verbose)
        return(cpy)

    def info(self, verbose=None):
        if verbose is None: verbose = self.verbose
        ushots = np.unique(self.da[self.mainkey])
        print('{nm} contains {ins:,}({mins:.1f}M) instances from {s} {mainkey}s'\
                  ', {ks} data arrays'
              .format(nm = self.name,
                      ins=len(self.da[self.mainkey]),
                      mins=len(self.da[self.mainkey])/1e6,
                      s=len(ushots), mainkey=self.mainkey,
                      ks = len(self.da.keys())))
        if len(ushots) < 10:
            shotstr=str('{s}'.format(Shots=self.mainkey,s=ushots))
        else: 
            shotstr = str('{s}...'.format(
                s=','.join([str(sh) for sh in ushots[0:3]])))
            shotstr += str('{s}'.format(
                s=','.join([str(sh) for sh in ushots[-4:]])))

        if verbose==0:
            # Shots is usually the main key, but allow for others
            print('{Shots} {s}, vars are {ks}'.format(Shots=self.mainkey, s=shotstr, ks=self.da.keys()))
        else:
            if (not self.loaded) and (self.len > 1e6): 
                print('may be faster to load first')

            print('{Shots} {s}\n Vars: '.format(Shots=self.mainkey,s=shotstr))

            lenshots = self.len
            for k in np.sort(list(self.da.keys())):
                varname = k
                var = self.da[k]
                if hasattr(var, 'dtype') and var.dtype=='O':
                    var = var.tolist()
                shp = np.shape(var)
                varname += str('[{s}]'
                               .format(s=','.join([str(i) for i in shp])))
                if len(shp)>1: 
                    varlen = len(var)
                    fac = np.product(shp[1:])
                else:
                    if hasattr(var, 'keys'):
                        varlen = len(var.keys())
                    else:    
                        varlen = len(var)

                    fac = 1  # factor to scale size - 
                             #e.g. second dimension of array

                # determine extent filled - faster to keep track of invalid entries
                if hasattr(var, 'dtype'):
                    typ = var.dtype
                    if np.issubdtype(var.dtype, np.int): 
                        minint = np.iinfo(var.dtype).min
                        invalid = np.where(var == minint)[0]
                    elif np.issubdtype(var.dtype, np.float): 
                        invalid = np.where(np.isnan(var))[0]

                    else:
                        invalid = []
                        print('no validity criterion for key "{k}", type {dt}'
                              .format(k=k, dt=var.dtype))
                else: 
                    typ = type(var)
                    try:
                        invalid = np.where(np.isnan(var))[0]
                    except:
                        print('validity can not be determined for '\
                                  'key {k}, type {dt}'
                              .format(k=k, dt=typ))
                        invalid = []
                print('{k:24s}: {pc:7.1f}%'.
                      format(k=varname, 
                             #pc = 100*(len(np.where(self.da[k]!=np.nan)[0])/
                             pc = 100*(1-(len(invalid)/float(lenshots)/fac)))),
                print(typ),
                if varlen != lenshots and k != 'info': 
                    print('Warning - array length {al} != shot length {s} '
                          .format(al=varlen,s=lenshots))
                else: print('')  # to close line
        print(self.name)

    def make_attributes(self):
        """ make each element of the dictionary an attribute of the DA object
        This is very convenient for operations on more than one dataset e.g.
        plot(da.im2 - daold.im2)
        Is this python 3 compatible?  It seems to work fine for continuum 3.5.1
        """
        for key in list(self):
            if hasattr(self, key):
                print('Not replacing attribute ' + key)
            else:
                exec("self."+key+"=self['" + key + "']")
  
        
    def load(self, sel=None):
        start_mem = report_mem(msg='load', verbose=self.verbose)
        st = seconds()
        if sel is None: # the arg overrides the object value
            sel = self.sel
        else:
            print('Overriding any decimation - selecting {s:,} instances'
                  .format(s=len(sel)))
            self.sel = sel

        if self.verbose>0: print('loading {nm}'.format(nm=self.name)), 
        if self.loaded:
            if self.verbose: ('print {nm} already loaded'.format(nm=self.name))

        else:
            dd = {}
            for k in self.da.keys():
                if sel is None:
                    dd.update({k: self.da[k]})
                else: # selective (decimated to limit)
                    try:
                        dd.update({k: self.da[k][sel]})
                    except Exception as reason:
                        dd.update({k: self.da[k]})
                        print('{k} loaded in full: {reason}'
                              .format(k=k,reason=reason))
                # dictionaries get stored as an object, need "to list"
                debug_(self.debug, 2, key='limit')
                if hasattr(dd[k],'dtype'):
                    if dd[k].dtype == np.dtype('object'):
                        dd[k] = dd[k].tolist()
                        if self.verbose > 0: 
                            print('object conversion for {k}'
                                  .format(k=k))
                    if (hasattr(dd[k],'dtype') and 
                        dd[k].dtype == np.dtype('object')):
                        dd[k] = dd[k].tolist()
                        print('*** Second object conversion for {k}!!!'
                              .format(k=k))
        # key 'info' should be replaced by the more up-to-date self. copy
        self.da = dd
        self.update({'info': self.infodict}, check=False)
        if self.verbose > -1: print(' in {dt:.1f} secs'.format(dt=seconds()-st))
        report_mem(start_mem, verbose=self.verbose)
        return(True)


    def save(self, filename, verbose=None, sel=None, use_dictionary=False,tempdir=None, zipopt=-1):
        """ Save as an npz file, using an incremental method, which
        only uses as much /tmp space as required by each var at a time.
        Select which to save with sel: if sel is None, save all except
        for use_dictionary below.
        If use_dictionary is a valid dictionary, save the values of
        ANY AND ONLY the LOCAL variables whose names are in the keys for
        this set.
        So if you have extracted a subset, and you specify 
        use_dictionary=locals(), only that subset is saved (both in array 
        length, and variables chosen).
        Beware locals that are not your variables - e.g. mtrand.beta
        To avoid running out of space on tmp, or to speed up zip - 
        Now included as an argument
        (Note that the normal os.putenv() doesn't seem to write to
        THIS environment use the fudge below - careful - no guarantees)
        os.environ.__setitem__('TMPDIR',os.getenv('HOME'))
        actually - this seems OK
        os.environ['IGETFILE']='/data/datamining/myView/bin/linux/igetfile'

        reload tempfile
        tempfile.gettempdir()
        also ('ZIPOPT','"-1"')  (Now incorporated into args, not tested)
        ** superseded by zlib.Z_DEFAULT_COMPRESSION 0--9  (or -1 for default)
        """
        if verbose is None: verbose = self.verbose
        st = seconds()

        if tempdir is not None:
            os.environ.__setitem__('TMPDIR', tempdir)
            import tempfile
            reload(tempfile) # in case it was already imported
            if tempfile.gettempdir() != tempdir:
                warn('failed to set tempdir = {t}: Value is {v}'
                     .format(t=tempdir, v=tempfile.gettempdir()))

        import zlib
        zlib.Z_DEFAULT_COMPRESSION = int(zipopt)
        #""" now obsolete zipfile calls zlin with Z_DEFAULT_COMPRESION arg
        #import os
        #print('overriding default zip compression to {z}'.format(z=zipopt))
        #os.environ.__setitem__('ZIPOPT', str(zipopt))
        #"""

        if use_dictionary == False: 
            save_dict = self.da # the dict used to get data
        else: 
            save_dict = use_dictionary
            print('Warning - saving only a subset')


        if sel is not None:
            use_keys = sel
        else:
            use_keys = []
            for k in self.da.keys():
                if k in save_dict.keys():
                    use_keys.append(k)

        if verbose: print(' Saving only {k}'.format(k=use_keys))

        if 'info' in use_keys:  info = info_to_bytes(self['info'])  # avoid py3 unicode error

        args=','.join(["{k}=save_dict['{k}']".
                       format(k=k) for k in use_keys])
        if verbose:
            print('lengths: {0} -999 indicates dodgy variable'
                   .format([mylen(save_dict[k]) for k in use_keys]))

        if 'mask' in self.da:
            self.infodict.update({'valid_keys': self.masked.valid_keys})
            self.da['info'] = np.array(self.infodict)
            self.update({'info': self.infodict}, check=False)

        if self.debug:
            print('saving '+filename, args)

        exec("np.savez_compressed(filename,"+args+")")
        self.name = filename

        if verbose: print(' in {dt:.1f} secs'.format(dt=seconds()-st))

    def extract(self, dictionary=False, varnames=None, inds=None, limit=None, strict=0, masked=1, debug=0):
        """ extract the listed variables into the dictionary (local by default)
        selecting those at indices <inds> (all be default
        variables must be strings, either an array, or separated by commas

        if the dictionary is False, return them in a tuple instead
        Note: returning a list requires you to make the order consistent

        if varnames is None - extract all.

        e.g. if da is a dictionary or arrays
        da = DA('mydata.npz')
        da.extract(varnames='shot,beta')
        plot(shot,beta)

        (shot,beta,n_e) = da.extract(varnames=['shot','beta','n_e'], \
                                      inds=np.where(da['beta']>3)[0])
        # makes a tuple of 3 arrays of data for high beta.  
        Note   syntax of where()! It is evaluted in your variable space.
               to extract one var, need trailing "," (tuple notation) e.g.
                    (allbeta,) = D54.extract(locals(), 'beta')
               which can be abbreviated to
                    allbeta, = D54.extract(locals(), 'beta')
        
        """
        start_mem = report_mem(msg='extract', verbose=self.verbose)
        if debug == 0: debug = self.debug
        if varnames is None: varnames = self.da.keys()  # all variables

        if isinstance(varnames, six.string_types):
            varlist = varnames.split(',')
        else: varlist = varnames
        val_tuple = ()

        if inds is None:
            inds = np.arange(self.len)
        if (len(np.shape(inds))==2): 
            inds = inds[0]   # trick to catch when you forget [0] on where

        if limit != None and len(inds)> abs(limit):
            if limit<0: 
                print('repeatably' ),
                np.random.seed(0)  # if positive, should be random
                                   # negative will repeat the sequence
            else: print('randomly'),
                
            print('decimating from sample of {n} and'.format(n=len(inds))),
            ir = np.where(np.random.random(len(inds))
                          < float(abs(limit))/len(inds))[0]
            inds = inds[ir]

        if len(inds)<500: print('*** {n} is a very small number to extract????'
                                .format(n=len(inds)))

        if self.verbose>0:
            print('extracting a sample of {n} '.format(n=len(inds)))

        for k in varlist:
            if k in self.da:  # be careful that self keys is up to date!
                                # this is normally OK if you use self.update
                debug_(debug,key='extract')
                # used to refer to da[k] twice - two reads if npz
                # if masking is enabled and it is a legitimate key for masking, use the masked version
                if masked and hasattr(self, 'masked') and k in self.masked.keys():
                    if self.verbose > -1:
                        print('extracting masked values for {k} - '
                              'use masked=0 to get raw values'.format(k=k))
                    dak = self.masked[k]
                else:
                    dak = self.da[k]      # We know we want it - let's
                                          #   hope space is not wasted
                if hasattr(dak, 'keys'):  # used to be self.da[k]
                    allvals = dak
                else:
                    allvals = np.array(dak)

                if len(np.shape(allvals)) == 0:
                    sel_vals = allvals
                else:
                    if (len(np.shape(allvals)) > 0) and (len(allvals) < np.max(inds)):
                        print('{k} does not match the size of the other arrays - extracting all'.format(k=k))
                        sel_vals = allvals
                    else:
                        sel_vals = allvals[inds]
                if dictionary is False:
                    val_tuple += (sel_vals,)
                else:
                    dictionary.update({k: sel_vals})
            else: print('variable {k} not found in {ks}'.
                        format(k=k, ks = np.sort(self.da.keys())))
        report_mem(start_mem, verbose=self.verbose)
        if dictionary is False:
            return(val_tuple)

    def hist(self, key, bins=50, nanval=-0.01, percentile=99, label=None):
        """ plot a histogram of Te or resid etc, replacing Nans or infs
        with nanval, and considering only up to the <percentile>th percentile
        DA('LP20160310_9_L57__amoeba21_1.2_2k.npz').hist('resid')

        Examples:
          >>> da.hist('resid',percentile=97,label='{k}: {fn}')
          >>> da.hist('resid',percentile=97,label='{k}: {actual_fit_params}')
          >>> da.hist('resid',percentile=97,label='{k}: {i_diag} {actual_fit_params}')
       """
        dat = self.da[key]
        wn = np.where(np.isinf(dat) | np.isnan(dat))
        dat[wn] = nanval
        print('{n} inf or NaNs excluded'.format(n=len(wn[0])))
        biginds = np.argsort(np.abs(dat).flatten())[::-1]
        # go for 99th percentile
        useinds = biginds[int((1-percentile/100.0)*len(biginds)):]
        if label is None:
            label = key
        elif '{' in label:
            try:
                label = label.format(k=key, fn=self.name, **self['info']['params'])
            except Exception as reason:
                print('hist label failed because {r}'.format(r=reason))
        plt.hist(dat.flatten()[useinds], bins=bins, label=label)
        sz = ['small','x-small'][len(label)>150]
        plt.legend(prop=dict(size=sz))
        plt.show()

    def plot(self, key, xkey='t', label_fmt='{lab}', sharey=1, select=None, inds=None, sharex='col', masked=1, ebar=1, marker='', elinewidth=0.3, ref_line=0, style=None, axlist=None, **kwargs):
        """ 
        Plot the member 'key' of the Dictionary of Arrays 
        kwargs:
          masked [1] 1 show only 'unmasked' points
          ebar [1]  the number of points per errorbar - None will suppress
          ref_line [None]  - draw a horizontal line at value, or draw None
          style - 'step' will use step plots (automatic for showing the mask)
          axlist a list of axes so that data can be overplotted - see below
          inds - an array to sort the plot order

        A mask of True will allow that point to be seen. 
        Examples:
          da.plot('Te') 
          da.plot('Te', ebar=10, marker='o')  # plot with 'o's, show every 10th error bar
          da.plot('mask')                     # to see mask = 1 == show, 0 is suppress

        # example of overplotting:
        from pyfusion.data.DA_datamining import Masked_DA, DA
        da=DA('LP/LP20171018_19_UTDU_2k8.npz')
        da21=DA('/tmp/LP20171018_19_UTDU_2k8_lsq_lpf21.npz')
        ax=da.plot('mask')            # new plot
        da21.plot('mask',axlist=ax)   # data overplotted for comparison

        """
        from matplotlib.ticker import MaxNLocator
        if sharey == 1:
            sharey = 'all'
        if xkey not in self and 't_mid' in self:
            print("no 't' available - defaulting to 't_mid'", end=', ')
            xkey = 't_mid'

        if masked and hasattr(self, 'masked') and key in self.masked.keys():
            print('using masked {k}'.format(k=key), end=', ')
            arr = self.masked[key]
        else:
            arr = self[key]

        if inds is None:
            inds = range(len(arr))

        style = 'step' if style is None and key == 'mask' else style
        print(style)
        x = np.array(self[xkey])
        nchans = np.shape(arr)[1]
        if nchans > 20: 
            print('warning- very many channels {c} - ^C to kill'
                  .format(c=nchans)) 
        if select is not None:
            clist = select
            nvis = len(select)
        else:
            nvis = nchans
            clist = range(nchans)

        if 'channels' in self.infodict:
            labs = self.infodict['channels']
        else:
            labs = [str(ch) for ch in range(nchans)]
        if axlist is not None:
            axs = axlist
            fig = plt.gcf()
        else:
            fig, axs = plt.subplots(nvis, 1, squeeze=1, sharey=sharey, sharex=sharex)
        if nvis == 1:
            axs = [axs]
        # 3 bins -> 4 ticks max
        locator = MaxNLocator(nbins=min(10, max(3, 20//nvis)), prune='upper')
        #  Allow default room (not much!) for x ticks if x-axis is not shared
        hspace = 0 if sharex == 'col' else None
        fig.subplots_adjust(top=0.95, hspace=hspace, bottom=0.05)
        for c, (ch, ax) in enumerate(zip(clist, axs)):
            # print(ch)
            (rot,ali) = ('vertical','center') if nvis < 6 else ('horizontal','right')
            ax.set_ylabel(labs[ch], rotation=rot,
                          horizontalalignment=ali)
            if np.isnan(arr[:, ch]).all():  # all nans confuses sharey
                ax.plot(x, x*0)
            # grab the common features to both plots in a dict to simplify
            kwargs.update(dict(marker=marker, label=label_fmt.format(lab=labs[ch])))
            if ebar is not None and 'e'+key in self.da:
                yerr = self.da['e'+key]
                kwargs.update(dict(errorevery=ebar, elinewidth=elinewidth, mew=elinewidth))
                ax.errorbar(x[inds], arr[inds, ch], yerr=yerr[inds, ch], **kwargs)
            else:
                plotter = ax.step if style == 'step' else ax.plot
                plotter(x[inds], arr[inds, ch], **kwargs)

            ax.yaxis.set_major_locator(locator)
            if ref_line is not None:
                ax.plot(ax.get_xlim(), [ref_line, ref_line], 'c', lw=0.5)
            # this worked in data/plots.py - but warning in DA_datamining
            # ax.locator_params(prune='both', axis=x)
            # but the plot doesnt overwrite axis labels for sharex='col' anyway
            """  This suppresses all if sharex='col'
            print(c+1, nvis)
            if (c+1 != nvis):
                ax.set_xticklabels('')
            """

        # plt.legend((prop=dict(size='small'))
        plt.suptitle(self.name.replace('.npz', '') + ' ' + key)
        debug_(self.debug, 2, key='DA_plot')
        plt.show(0)
        return(axs)   # the returned array allows overplotting on each axis

def da(filename='300_small.npz',dd=True):
    """ return a da dictionary (used to be called dd - not the DA object)
    mainly for automated tests of example files.
    """
    if dd: return(DA(filename,load=1).da)
    else: return(DA(filename,load=1))

if __name__ == "__main__":

    print('Running a few small tests')
    d=dict(shot= [1,2,3], val=[1.,2.,3])
    da = DA(d)
    da.extract(locals(),'shot,val')
    print(shot, val)

    # ???? gives error when run a second time??? bug?
    tfile = '/tmp/junk'
    da.save(tfile)
    append_to_DA_file(tfile, dict(A=[3,4,5]))
    da_bigger = DA(tfile)
    da_bigger.info(2)
