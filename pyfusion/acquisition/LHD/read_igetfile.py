from numpy import sin, cos, shape, average, array, max, abs, mod
import os
import pyfusion
import subprocess
from pyfusion.debug_ import debug_
from time import sleep

""" get data from the "routine diagnostics" server, using igetfile
The data comes in a text form, self describing.  The class "igetfile"
  igetfile -s shot -d wp  -> wp@99999.dat
  Example for data locally stored in a file
     ts=igetfile('cache/TS{0:06d}.dat.bz2',shot=36248)
     ts.plot() 
  read_igetfile is procedural and is used by the class.  plotting
  in the class version is nicer than for the procedural version.

""" 

colorset = ('b,g,r,c,m,y,k,orange,purple,lightgreen,gray,brown,teal,tan'.split(','))
def call_igetfile(path_to_igetfile, filename):
    """ run the igetfile program, returning the filename, which can be
    used for later deleting.
    assume filename looks like somedir/wp@90000.dat
    """
    
    filepart = filename.split('/')[-1]
    (name, rest) = filepart.split('@')
    shot = int(rest.split('.')[0])
    cmd = str("{ptoexe} -s {sh} -d {diag} -o {fn}"
              .format(ptoexe = path_to_igetfile,
                      sh = shot, diag = name, fn = filename))

    if (pyfusion.VERBOSE > 1): print('IGETFILE: %s' % (cmd))

    attempt = 1
    while(1):
         if attempt>1: print('attempt {a}, {c}'.format(a=attempt, c=cmd))
         retr_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
         (resp,err) = retr_pipe.communicate()
         if (resp != "") or (retr_pipe.returncode != 0): 
              print("IGETFILE Error code {ecode}, err = {err}, \n"
                    "on attempt {att}, stdout => {resp}"
                    .format(ecode=retr_pipe.returncode,resp=resp,
                            err=err,att=attempt))
              if attempt>10 or ("not exist" in err) or ( "data not found" in resp): 
                   raise LookupError(str("Error %d accessing igetfile: cmd=%s \n"
                                         "stdout=%s, stderr=%s" % 
                                         (retr_pipe.poll(), cmd, resp, err)))
              attempt += 1
              sleep(2)
         else:
              break
         
    return(filename)



class igetfile():
    data = None
    vardict = {}
    shot = None
    filename = ""
    

    def __init__(self, filename=None, fileformat=None, shot=None, verbose=0, plot=False, debug=0, hold=True):
        """ read in data, optionally plot (True gets default channel, N gets ch=N
        if filename contains a {, assume it is a format
        let get_basic_params deal with .bz2 files etc for now.
        """
        self.debug = debug
        if filename is None or "{" in filename:
            self.shot = int(shot)
            if "{" in filename: self.filename = filename.format(self.shot)
            else: self.filename = fileformat.format(self.shot)
        else:
            self.filename = filename

        path_to_igetfile = os.getenv('IGETFILE') 
        if path_to_igetfile != None: # returns None starting 2.6
            self.filename = call_igetfile(path_to_igetfile, self.filename)
                                     
        (self.data,self.vardict) = read_igetfile(filename=self.filename, 
                                                 verbose=0, plot=plot, hold=hold,debug=debug)
        self.valnames=self.vardict['ValName'] # shortcut
        # get rid of file if we got it from the server
        if path_to_igetfile != None and pyfusion.DEBUG<3:
            os.remove(self.filename)

        if plot !=0: 
            if type(plot) == type(True):  self.plot()
            else: self.plot(ch=plot)

    def plot(self, hold=0, ch=-1, tstart=None, tend=None, tstep=None, scl=None, navg=1, debug=1, axes=None, *args, **kwargs):
        """ ch=-1 -> Plot all the data items - can take a long time.
        """
        import pylab as pl
        name=self.vardict['Name']
        dim=self.vardict['DimSize']
        nv=self.vardict['ValNo']
        nd=self.vardict['DimNo']
        if ch is None: 
            if name =='lhd_mse1':
                ch = 19
            elif name.lower() =='thomson':
                ch=3
            else:
                ch = 1

        if axes != None: ax = axes
        else: ax = pl.gca()

        ax.plot(hold=hold)
        
        linest = ['-',':','--','-.']
        if pyfusion.DEBUG>0: print('shape is {s}'.format(s=shape(dim)))
        if len(shape(dim))==0:
            for (p,name) in enumerate(self.vardict['ValName']):
                col, lin = divmod(p, len(linest))
                if ch==-1 or ch==(p+1):
                    ax.plot(self.data[:,0], self.data[:,p+1],
                            label="{0}:{1}".format(name, self.shot),
                            linestyle=linest[lin], 
                            color = colorset[col % len(colorset)],
                            *args, **kwargs)
                
        else:  # 2 dim plots
            # reshape into radial profiles
            data3D=self.data.reshape(dim[0],dim[1],nd+nv)
            if tstart is None: tstart=0
            if tend is None: tend=dim[0]/2
            if tstep is None: tstep=max([tend/50,1])  # reduce to 50 steps
            # check maximum to determine scaling
            if scl is None: scl = max(abs((data3D[:,:,ch+nd-1])/50))  # was /10
            if debug>0: 
                print("tstart=%.4g, tstep=%.4g, tend=%.4g, scl=%.4g))" % 
                      (tstart, tstep, tend, scl))
            # if more than 10 profiles, plot all in grey, then
            if (((tend-tstart)/tstep) > 10) and tstep>0: 
                for t in range(tstart,tend,tstep): 
                    xval = average(data3D[t:t+navg,:,nd-1],0)
                    ax.plot(xval, scl*t+average(data3D[t:t+navg,:,ch+nd-1],0),
                            color=0.8-(0.4/5)*mod(t-tstart,5)*array([1,1,1]))

            if navg>1: avg="avg%d" % (navg)
            else: avg = ''
            # plot every 5th one in colour

            for t in range(tstart,tend,tstep*5): 
                lab=avg+"%s:%.4g %s" % (self.vardict['ValName'][ch-1],data3D[t,0,0],self.vardict['DimUnit'][0])
                ax.plot(xval,scl*t+average(data3D[t:t+navg,:,ch+nd-1],0),
                        label=lab,*args, **kwargs)

        square_inches = (array(ax.get_figure().get_size_inches()).prod() /
                         array(ax.get_geometry()[0:2]).prod())
        if square_inches< 15: pl.rcParams['legend.fontsize']='small'
        if square_inches > 5: ax.legend()
        pl.title("%d:%s" % (self.vardict['ShotNo'],self.vardict['Name']))

def read_igetfile(filename=None, verbose=0, plot=True, hold=True, debug=0, quiet=0):
    """ Read LHD data from igetfile
    Note that the plot option on this call is crude, compared to in igetfile
        pseudocode: 
           read file into a buffer
           find "[Parameters]"
           find "[data]"
           put brackets around lines with = sign and unquoted comma
           read stuff between
           read data into a rank 2 array

        ( This example creates it's own file-like object - not normally
        required - also note that we use \\n (double backslash.
        Finally, this test runs into a problem at the first line of data
        maybe new since 2.7?)

        >>> from StringIO import StringIO
        >>> data = ' \\n'.join(["%f, %f, %f" % (n/1e3, sin(n/1e3), cos(n/1e3)) for n in range(4)])
        >>> st=StringIO("#[Parameters]\\n# Name = 'mmw'\\nShotNo=50623\\n# Dimno=1 \\n " \
        + "# DimName = 'Time'\\n# DimSize = 2600\\n# DimUnit = 's'\\n# \\n" \
        + "# ValNo = 2\\n# ValName = 'nL()', 'nL()'\\n# ValUnit = 'e19m-2', 'e19m-2'\\n" \
        + "# [Comments]\\n#  \\n# [data]\\n" \
        + data+" \\n ")
        >>> read_igetfile(st)[0][2]
        array([ 0.002   ,  0.002   ,  0.999998])


    """
    from numpy import where, diff, loadtxt, array
    if (quiet > 0) and (debug > 0): quiet=0  # debug is stronger than quiet
    import time
    from StringIO import StringIO

    if filename is None: filename = '/home/bdb112/python/mmw@50623.dat'
    if os.path.split(filename)[0] == '':
        pref,rest = filename.split('@')
        filename = os.path.join(pyfusion.config.get('global','local_diag_path'), pref, filename)
        # look for igetfile data in a few places
        debug_(max(pyfusion.DEBUG, debug), level=4, key='find_data')
    for ext in ['', '.gz', '.bz2']:
        if os.path.isfile(filename+ext):
            filename += ext
            break
    else:  #this goes with the for - if it completes without a break
        raise IOError('igetfile {f} or .gz, .bz2 not found.'.format(f=filename))

    strt = time.time()
#    print(filename.readlines())
#    print('asd************8')
#    filename.seek(0)
#    return([1,2,3])
    try:  # this one is to catch missing files
        try:
            # loadtxt does auto decompression~ 3x slower than uncompressed on lenny
            # avoid breaking into token by using impossible delimiter
            # or just "\n" will do
            #   comments has to be a character - '\000' seems OK
            fbuff = loadtxt(filename, delimiter='\n', dtype='S',comments='~')
            # careful - loadtxt removes the \n, readlines doesn't
            reader='loadtxt'
        except:     
            if type(filename) == type('str'): fp=open(filename)
            else: fp=filename
            fp.seek(0)
            #print(fp.readlines())
            #return([1,2,3,4])
            fbuff=fp.readlines()
            reader='readlines'
    except Exception as info:
        if quiet>0: 
            pass
        else:
            raise IOError('{f} not found: {info} {a}'.format(f=filename,info=info,a=info.args))

    if verbose>0: print ("Used %s" % reader), 
    pinds = [st.upper().find("[PARAMETERS]") for st in fbuff ]
    pstarts = where(array(pinds)>-1)[0]   # where function only operates on arrays!
    cinds = [st.upper().find("[COMMENTS]") for st in fbuff ]
    cstarts = where(array(cinds)>-1)[0]   # where function only operates on arrays!
    dinds = [st.upper().find("[DATA]") for st in fbuff ]
    dstarts = where(array(dinds)>-1)[0]
    if verbose: print(", %d data lines found" % (len(dstarts))),

    trystart=0
    found=False
    from time import sleep
    while not(found) and trystart<len(pstarts):
        ValName = None
        vardict = {}
        parmlines = fbuff[pstarts[trystart]+1:cstarts[trystart]]
        parms = [line[2:].strip() for line in parmlines]
        for (p,parm) in enumerate(parms):
            if verbose>1: print(parm)
            if parm.find(',')>=0: parms[p] = '= ['.join(parm.split('=')) + ']'
            try:
                # this tries to execute vvv=VVV putting the result in vardict
                # syntax is exec(str, globaldict, localdict)
                exec(parms[p],{},vardict)
            except SyntaxError:
                if parm.find(',')>0:
                    print('bad syntax {%s} try to reparse' % (parms[p])),
                    (lhs,rhs) = parm.split('=')
                    parsesyms = rhs.replace("'","").split(",")
                    reparse = (lhs+"=["
                               + ','.join(["'%s'" % (sym.strip()) for sym in parsesyms])
                               + ']')
                    try:
                        exec(reparse,{},vardict)                               
                        print(': recovered {%s}' % reparse)
                    except:
                        print('Error executing reparsed [%s]' % (reparse))

            except:
                print('Error executing [%s]' % (parms[p]))
               
#        if ValName != None:
        if 'ValName' in vardict:
            if verbose: print('Found %s ' % (ValName))
            found = True
            
    if not(found): 
        raise LookupError(' data not in %s ' % (filename))


# StringIO allows a string to be treated like a file.
# could alternatively pass a skiprows to loadtxt - but this avoids re-reading file

# the '\n' is needed when the buffer comes from loadtxt.  I hope the next loadtxt doesn't mind.
    sio = StringIO('\n'.join(fbuff[dstarts[trystart]+1:]))
    ## this is an attempt to debug the test code - 
    """ x=sio.readlines()
    print([ord(c) for c in x[0]])
    print(x[0])
    print(x[1])
    sio.seek(0)
    return(sio.readlines())
    """
    arr=loadtxt(sio,delimiter=',') 
    if plot: 
        import pylab as pl
        pl.plot(arr[:,0],arr[:,1],'.',markersize=1, hold=hold)
    return(arr,vardict)
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
"""
#self=igetfile(filename='/home/bdb112/LHD/TS099000.dat')
import os
import pylab as pl
#print(os.path.split(???__str__.split("'")[3])[0]+'/TS099000.dat')
filename='/home/bdb112/pyfusion/pyfusion/acquisition/LHD/TS090000.dat.bz2'
self=igetfile(filename)
dim=self.vardict['DimSize']
nv=self.vardict['ValNo']
nd=self.vardict['DimNo']
data3D=self.data.reshape(dim[0],dim[1],nd+nv)
for t in range(0,dim[0],5): plot(average(data3D[t:t+10,:,4],0))
#for (t,dat) in enumerate(data3D[:,:]): print(dat[0])
tend=dim[0]/2
for t in range(0,tend,5): pl.plot(10*t+average(data3D[t:t+10,:,4],0),color=0.8*array([1,1,1]))

for t in range(0,tend,25): pl.plot(10*t+average(data3D[t:t+10,:,4],0),label=("%d ms" % (data3D[t,0,0])))

pl.legend()

"""

