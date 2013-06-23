import numpy as np
import pylab as pl
from time import time as seconds


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
        pm = psutil.avail_phymem()
        vm = psutil.avail_virtmem()
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
    

class DA():
    """ class to handle and save data in a dictionary of arrays
    can deal with databases larger than memory, by using load = 0
    faster to use if load=1, but if you subselect by using extract
    you get the speed for large data sets (once extract is done.
    Extract can be used over and over to get different data sets.
    """
    def __init__(self, fileordict, debug=0, verbose=0, load=0):
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
            if load == 0:
                self.loaded = 0
            else:
                self.loaded = self.load()

        self.keys = self.da.keys()
        if 'dd' in self.keys:  # old style, all in one
            print('old "dd" object style file')
            self.da = self.da['dd'].tolist()
            self.keys = self.da.keys()

        if 'shot' in self.keys:
            self.mainkey = 'shot'
        else:
            self.mainkey = self.da.keys()[0]

        self.len = len(self.da[self.mainkey])

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
                  .format(dl=len(dd['shot']), tl = self.len))

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
        return(deepcopy(self.da))

    def info(self, verbose=None):
        if verbose == None: verbose = self.verbose
        shots = np.unique(self.da[self.mainkey])
        print('{nm} contains {ins}({mins:.1f}M) instances from {s} shots'\
                  ', {ks} data arrays'
              .format(nm = self.name,
                      ins=len(self.da['shot']),
                      mins=len(self.da['shot'])/1e6,
                      s=len(shots),
                      ks = len(self.da.keys())))
        if verbose==0:
            print('Shots {s}, vars are {ks}'.format(s=shots, ks=self.da.keys()))
        else:
            if (not self.loaded) and (self.len > 1e6): 
                print('may be faster to load first')

            print('Shots {s}\n Vars: '.format(s=shots))
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

    def load(self):
        start_mem = report_mem(msg='load')
        st = seconds()
        if self.verbose>0: print('loading {nm}'.format(nm=self.name)), 
        if self.loaded:
            if self.verbose: ('print {nm} already loaded'.format(nm=self.name))

        else:
            dd = {}
            for k in self.da.keys():
                dd.update({k: self.da[k]})
                # dictionaries get stored as an object, need "to list"
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

        if self.verbose: print(' in {dt:.1f} secs'.format(dt=seconds()-st))
        self.da = dd
        report_mem(start_mem)

    def save(self, filename, verbose=None, use_dictionary=False):
        """ Save as an npz file, using an incremental method, which
        only uses as much /tmp space as required by each var at a time.
        if use_dictionary is a valid dictionary, save the values of
        ANY AND ONLY the LOCAL variables whose names are in the keys for
        this set.
        So if you have extracted a subset, and you specify 
        use_dictionary=locals(),
        only that subset is saved (both in array length, and variables chosen).
        Beware locals that are not your variables - e.g. mtrand.beta
        """
        if verbose == None: verbose = self.verbose
        st = seconds()
        if use_dictionary == False: 
            save_dict = self.da # the dict used to get data
        else: 
            save_dict = use_dictionary
            print('Warning - saving only a subset')


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
                if hasattr(self.da[k],'keys'):
                    allvals = self.da[k]
                else:
                    allvals = np.array(self.da[k])

                if len(np.shape(allvals)) == 0:
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
