import numpy as np
import pylab as pl
from time import time as seconds
from warnings import warn

""" Note: in programming, be careful not to refer to .da[k] unnecessarily
if it is not loaded - typically, if you plan to test it thne use it,
save to a var first, then test, then use it.  see "dak" below
"""

# first, arrange a debug one way or another
try: 
    from pyfusion.debug_ import debug_
except:
    def debug_(debug, msg='', *args, **kwargs):
        print('attempt to debug ' + msg + 
              " need boyd's debug_.py to debug properly")

def mylen(ob):
    """ return the length of an array or dictionary for diagnostic info """
    if type(ob) == type({}):
        return(len(ob.keys()))
    elif ob == None:
        return(0)
    else:
        try:
            return(len(ob))
        except:
            return(-999)
# try to use psutils to watch memory usage - be quiet if it fails
try:
    import psutil

    def report_mem(prev_values=None,msg=None):
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
        print('{msg}{pm:.2g} GB phys mem, {vm:.2g} GB virt mem avail'
              .format(msg=msg, pm=pm/1e9, vm=vm/1e9)),

        if prev_values is None:
            print
        else:
            print('- dt={dt:.2g}s, used {pm:.2g} GB phys, {vm:.2g} GB virt'
                  .format(pm=(prev_values[0] - pm)/1e9,
                          vm=(prev_values[1] - vm)/1e9, dt = tim-prev_values[2]))
        return((pm,vm,tim))
except None:
    def report_mem(prev_values=None, msg=None):
        return((None, None))
    
def append_to_DA(filename, dic, force=False):
    """ open filename with mode=a, after checking if the indx variables align
    force=1 ignores checks for consistent length c.f. the var shot
    Not a member of the class DA, because the class has memory copies of the
    file, so it would be confusing.
    """
    import zipfile, os
    import pyfusion
    dd = np.load(filename)
    error = None
    if 'indx' in dd.keys() and 'indx' in dic.keys():
        check = dd['indx']
        if (dic['indx']!=check).all():
            error=('mismatched indx vars in ' + filename)

    else:
        print('indx missing from one or both dictionaries - just checking size')
        check=dd['shot']
        if (len(check) != len(dic[dic.keys()[0]])):
            error=('mismatched var lengths in ' + filename)

    if error is not None:
        if force:  warn(error)
        else:      raise ValueError(error)

    zf = zipfile.ZipFile(filename,mode='a')
    tmp =  pyfusion.config.get('global', 'my_tmp')
    for key in dd.keys():
        tfile = tmp+'/'+ key+'.npy'
        np.save(tfile, dd[key])
        zf.write(tfile,arcname=key)
        os.remove(tfile)
    zf.close()

class DA():
    """ class to handle and save data in a dictionary of arrays
    can deal with databases larger than memory, by using load = 0
    faster to use if load=1, but if you subselect by using extract
    you get the speed for large data sets (once extract is done.
    Extract can be used over and over to get different data sets.

    mainkey is not necessarily a unique identifier - e.g it can be shot.
    decimate (via limit=) here applies at the load into memory stage (in self.da) - it 
    is the most effective space saver, but you need to reload if more data
    is needed.  The alternative is to decimate at the extract stage (but this
    applies only to the variables extracted into namespace (e.g. locals())
    """
    def __init__(self, fileordict, debug=0, verbose=0, load=0, limit=None, mainkey=None):
        # may want to make into arrays here...
        self.debug = debug
        self.verbose = verbose
        self.loaded = False
        
        start_mem = report_mem(msg='init')
        if (type(fileordict) == dict) or hasattr(fileordict,'zip'): 
            self.da = fileordict
            self.loaded = True
            self.name = 'dict'
        else:
            self.da = np.load(fileordict)
            self.name = fileordict
            self.loaded = 0

        self.keys = self.da.keys()
        if 'dd' in self.keys:  # old style, all in one
            print('old "dd" object style file')
            self.da = self.da['dd'].tolist()
            self.keys = self.da.keys()

        # make the info available to self.
        # needs different actions if npzfile and not yet loaded
        # we don't load yet, as we may want to decimate
        if 'info' in self.keys:
            if (hasattr(self.da['info'],'dtype') and                 self.da['info'].dtype == np.dtype('object')):
                self.infodict = self.da['info'].tolist()
            else:
                self.infodict = self.da['info']
        else:
            self.infodict = {}

        self.mainkey = mainkey  # may be None!
        debug_(self.debug, 2)
        if self.mainkey is None:
            if 'mainkey' in self.infodict.keys():
                self.mainkey = self.infodict['mainkey']
            else:
                if 'shot' in self.keys:
                    self.mainkey = 'shot'
                else:
                    self.mainkey = self.da.keys()[0]

        self.infodict.update({'mainkey':self.mainkey}) # update in case it has changed
        if type(self.da) == dict:
            self.da.update({'info': self.infodict})
        
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

        if load == 0:
            self.loaded = 0
        else:
            self.loaded = self.load()

        start_mem = report_mem(start_mem)
    #shallow_copy = try:if da.copy

    def append(self, dd):
        """ append the data in dd to the data in self

        """
        for k in self.da.keys():
            if k not in dd.keys():
                raise LookupError('key {k} not in dd keys: {keys}'
                                  .format(k=k, keys=dd.keys()))
        for k in dd.keys():
            if k not in self.da.keys():
                raise LookupError('key {k} not in DA keys: {keys}'
                                  .format(k=k, keys=self.da.keys()))
        for k in self.da.keys():
            if hasattr(dd[k],'keys'):
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

    def to_sqlalchemy(self,db = 'sqlite:///:memory:',n_recs=1000, chunk=1000):
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
        """ FOrmat for mysql load data infile
\N	27153	0.1	-1	0.285	11.7753	1	10010	-0.000908817	2.75	0.074	18.3	0.0156897	-0.118291	1.25	27153	\N	-1	-1	-1	\N	0.373338	\N	-0.00219764	0.001536	\N  etc
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
        self.fs_table = SA.Table('fs_table', self.metadata)
        (dbkeys,dbtypes)=([],[])
        for k in self.da.keys():
            arr = self.da[k]
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
                    self.metadata.tables['fs_table'].append_column(SA.Column(k, typ))
                    debug_(self.debug, 1)

            if self.debug>0: print(self.metadata.tables)

        if len(dbkeys)==0: return('nothing to create')
        self.metadata.create_all(self.engine)
        conn=self.engine.connect()
        if self.len > n_recs: print('Warning - only storing n_rec = {n} records'
                                    .format(n=n_recs))
        for c in range(0,min(n_recs,len(self.da[dbkeys[0]])),chunk):
            print(c, min(c+chunk, self.len))
            lst = []
            for i in range(c,min(c+chunk, min(self.len,n_recs))):
                dct = {}
                for (k,key) in enumerate(dbkeys): 
                    dct.update({key: cvt(self.da[key][i])})
                lst.append(dct)    
            if self.debug>0: print(lst)
            conn.execute(self.fs_table.insert(),lst)
                
                

    def copyda(self, force = False):
        """ make a deepcopy of self.da
        typically dd=DAxx.copy()
        instead of dd=DAxx.da - which will make dd and DAxx.da the same thing (not usually desirable)
        """
        from copy import deepcopy
        if (not force) and (self.len > 1e7): 
            quest = str('{n:,} elements - do you really want to copy? Y/^C to stop'
                        .format(n=self.len))
            if pl.isinteractive():
                if 'Y' not in raw_input(quest).upper():
                    1/0
            else: 
                print(quest)

        if self.loaded == 0: self.load()
        start_mem = report_mem(msg='copying')
        cpy = deepcopy(self.da)
        report_mem(start_mem)
        return(cpy)

    def info(self, verbose=None):
        if verbose == None: verbose = self.verbose
        shots = np.unique(self.da[self.mainkey])
        print('{nm} contains {ins}({mins:.1f}M) instances from {s} {mainkey}s'\
                  ', {ks} data arrays'
              .format(nm = self.name,
                      ins=len(self.da[self.mainkey]),
                      mins=len(self.da[self.mainkey])/1e6,
                      s=len(shots), mainkey=self.mainkey,
                      ks = len(self.da.keys())))
        if verbose==0:
            # Shots is usually the main key, but allow for others
            print('{Shots} {s}, vars are {ks}'.format(Shots=self.mainkey, s=shots, ks=self.da.keys()))
        else:
            if (not self.loaded) and (self.len > 1e6): 
                print('may be faster to load first')

            print('{Shots} {s}\n Vars: '.format(Shots=self.mainkey,s=shots))
            lenshots = self.len
            for k in np.sort(self.da.keys()):
                varname = k
                var = self.da[k]
                shp = np.shape(var)
                varname += str('[{s}]'
                               .format(s=','.join([str(i) for i in shp])))
                if len(shp)>1: 
                    varlen = len(var[:,1])
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

    def load(self, sel=None):
        start_mem = report_mem(msg='load')
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
                    except Exception, reason:
                        dd.update({k: self.da[k]})
                        print('{k} loaded in full: {reason}'
                              .format(k=k,reason=reason))
                # dictionaries get stored as an object, need "to list"
                debug_(self.debug, 1, key='limit')
                if hasattr(dd[k],'dtype'):
                    if dd[k].dtype == np.dtype('object'):
                        dd[k] = dd[k].tolist()
                        if self.verbose: 
                            print('object conversion for {k}'
                                  .format(k=k))
                    if (hasattr(dd[k],'dtype') and 
                        dd[k].dtype == np.dtype('object')):
                        dd[k] = dd[k].tolist()
                        print('*** Second object conversion for {k}!!!'
                              .format(k=k))
        # key 'info' should be replaced by the more up-to-date self. copy
        dd.update({'info': self.infodict})
        if self.verbose: print(' in {dt:.1f} secs'.format(dt=seconds()-st))
        self.da = dd
        report_mem(start_mem)

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
        reload tempfile
        tempfile.gettempdir()
        also ('ZIPOPT','"-1"')  (Now incorporated into args, not tested)
        ** superseded by zlib.Z_DEFAULT_COMPRESSION 0--9  (or -1 for default)
        """
        if verbose == None: verbose = self.verbose
        st = seconds()

        if tempdir is not None:
            import os
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


        args=','.join(["{k}=save_dict['{k}']".
                       format(k=k) for k in use_keys])
        if verbose:
            print('lengths: {0} -999 indicates dodgy variable'
                   .format([mylen(save_dict[k]) for k in use_keys]))

        exec("np.savez_compressed(filename,"+args+")")
        self.name = filename

        if verbose: print(' in {dt:.1f} secs'.format(dt=seconds()-st))

    def extract(self, dictionary = False, varnames=None, inds = None, limit=None,strict=0, debug=0):
        """ extract the listed variables into the dictionary (local by default)
        selecting those at indices <inds> (all be default
        variables must be strings, either an array, or separated by commas
        
        if the dictionary is False, return them in a tuple instead 
        Note: returning a list requires you to make the order consistent

        if varnames is None - extract all.

        e.g. if da is a dictionary or arrays
        da = DA('mydata.npz')
        da.extract('shot,beta')
        plot(shot,beta)

        (shot,beta,n_e) = da.extract(['shot','beta','n_e'], \
                                      inds=np.where(da['beta']>3)[0])
        # makes a tuple of 3 arrays of data for high beta.  
        Note   syntax of where()! It is evaluted in your variable space.
               to extract one var, need trailing "," (tuple notation) e.g.
                    (allbeta,) = D54.extract('beta',locals())
               which can be abbreviated to
                    allbeta, = D54.extract('beta',locals())
        
        """
        start_mem = report_mem(msg='extract')
        if debug == 0: debug = self.debug
        if varnames == None: varnames = self.da.keys()  # all variables

        if pl.is_string_like(varnames):
            varlist = varnames.split(',')
        else: varlist = varnames
        val_tuple = ()

        if inds == None:
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
            if k in self.keys:
                debug_(debug,key='extract')
                # used to refer to da[k] twice - two reads if npz
                dak = self.da[k]  # we know we want it - let's 
                                  # hope space is not wasted
                if hasattr(dak,'keys'): # used to be self.da[k]
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
                if dictionary == False: 
                    val_tuple += (sel_vals,)
                else:
                    dictionary.update({k: sel_vals})
            else: print('variable {k} not found in {ks}'.
                        format(k=k, ks = np.sort(self.da.keys())))
        report_mem(start_mem)
        if dictionary == False: 
            return(val_tuple)

if __name__ == "__main__":

    d=dict(shot= [1,2,3], val=[1.,2.,3])
    da = DA(d)
    da.extract(locals(),'shot,val')
    print(shot, val)
