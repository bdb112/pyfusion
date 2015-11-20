""" 
Originally a script in example (called get_basic_params - still there
get the basic plasma params for a given shot and range of times
interpolation overhead only begins at 10k points, doubles time at 1Million!!

"""

import pyfusion
import pylab as pl
import numpy as np
import os
import pylab as pl
from pyfusion.debug_ import debug_
from pyfusion.acquisition.LHD.read_igetfile import igetfile

from matplotlib.mlab import stineman_interp
from pyfusion.utils.read_csv_data import read_csv_data
from pyfusion.utils.utils import warn
import re

this_file = os.path.abspath( __file__ )
this_dir = os.path.split(this_file)[0]
acq_LHD = this_dir   #  from when it was in examples      + '/../acquisition/LHD/'
localigetfilepath=pyfusion.config.get('Acquisition:LHD','localigetfilepath')+'/'

""" Make a list of diagnostics, and how to obtain them, as a dictionary of dictionaries:
Top level keys are the short names for the dignostics
Next level keys are 
   format:    format string to generate file name from shot.
   name: wild card name to find in the file header
 For the case of static parameters, the 'format' field is a csv file containing them all.
"""
file_info={}
#file_info.update({'n_e': {'format': 'fircall@{0}.dat','name':'ne_bar(3939)'}})
# this form should be phase out, as it is an unsuitable variable name 
file_info.update({'<n_e19>': {'format': 'fircall@{0}.dat','name':'ne_bar\(3[0-9]+\)'}})
# fircall starts around 36142 and goes to at least 54198 - although 90000 has firc.
file_info.update({'n_e19b0': 
                  {'format': 'fircall@{0}.dat','name':'ne_bar\(3[0-9]+\)',
                   'comment': 'central line avg density'}})
# this one is an average of two ~ mid radius chords - simple way
# to reduce dependence on major axis position.
file_info.update({'n_e19dL5': 
                  {'format': 'fircall@{0}.dat',
                   'name':'average:nL\(4029|3399\)',
                   'comment': 'mid_radius ne_dL, not nebar'}})
file_info.update({'i_p': {'format': 'ip@{0}.dat','name':'Ip$'}})
file_info.update({'di_pdt': {'format': 'ip@{0}.dat','name':'ddt:Ip$'}})
file_info.update({'w_p': {'format': 'wp@{0}.dat','name':'Wp$'}})
file_info.update({'dw_pdt': {'format': 'wp@{0}.dat','name':'ddt:Wp$'}})
file_info.update({'dw_pdt2': {'format': 'wp@{0}.dat','name':'ddt2:Wp$'}})
file_info.update({'beta': {'format': 'wp@{0}.dat','name':'<beta-dia>'}})
# syntax is full regexp - so need .* not *
file_info.update({'NBI': {'format': 'nbi@{0}.dat','name':'sum:NBI.*.Iacc.'}})
file_info.update({'ech': {'format': 'ech@{0}.dat','name':'sum:.*'}})
file_info.update({'b_0': {'format': 'lhd_summary_data.csv','name':'MagneticField'}})
file_info.update({'R_ax': {'format': 'lhd_summary_data.csv','name':'MagneticAxis'}})
file_info.update({'Quad': {'format': 'lhd_summary_data.csv','name':'Quadruple'}})
file_info.update({'Gamma': {'format': 'lhd_summary_data.csv','name':'GAMMA'}})
file_info.update({'NBI1Pwr': {'format': 'lhd_summary_data.csv','name':'NBI1Power'}})
file_info.update({'NBI2Pwr': {'format': 'lhd_summary_data.csv','name':'NBI2Power'}})

global lhd_summary

def get_flat_top(shot=54196, times=None, smooth_dt = None, maxddw = None, hold=0, debug=0):
    """ debug=1 gives a plot
    """
    if times is None: times=np.linspace(0.02,8,8020) ;  
    from pyfusion.data.signal_processing import smooth

    bp=get_basic_diagnostics(shot=shot,diags=['w_p','dw_pdt','b_0'],times=times)
    # assume sign is OK - at the moment, the code to fix sign is in merge
    # but it is inactive.  Probably should be in get_basic_diag..
    # so far, it seems that w_p and i_p are corrected - not sure 
    # about other flux loops.
    w_p = bp['w_p']
    dw = bp['dw_pdt']
    w=np.where(w_p < 1e6)[0]  # I guess this is to exclude nans
    len(w)
    cent = np.sum(w_p[w]*times[w])/np.sum(w_p[w])
    icent = np.where(times > cent)[0][0]
    print("centroid = {0:.1f}".format(cent))
    if maxddw is None: maxddw = 100
    if smooth_dt is None: smooth_dt = 0.1 # smooth for 0.05 sec
    dt = (times[1]-times[0])
    ns = int(smooth_dt/dt)
    smootharr = [ns,ns,ns]
    offs = len(smootharr*ns)  # correction for smoothing offset
    dwsm = smooth(dw,n_smooth=smootharr)  # smooth dwdt
    ddw = np.diff(dwsm)/dt  #second deriv
    # work away from the centroid until 2nd deriv exceeds maxddw
    # assume 100kJ /sec is ramp, and a change of this over a second

    wb = int(0.5*offs) + np.nanargmax(dwsm)
    we = int(0.1*offs) + np.nanargmin(dwsm) # 
    wpmax = np.nanmax(w_p)
    # used to be maxddw - too restrictive now try dwsm
    wgtrev = np.where(np.abs(dwsm[icent-offs/2::-1])> maxddw*wpmax/100)[0]
    wgtfor = np.where(np.abs(dwsm[icent-offs/2:])> maxddw*wpmax/100)[0]
    if (len(wgtrev) < 10) or (len(wgtfor) < 10): 
        print('*** flat_top not found on shot {s}'.format(s=shot))
        return (0,0,(0,0,0,0,0))

    wbf = icent - wgtrev[0]
    wef = icent + wgtfor[0]
    if debug>0:
        pl.plot(w_p,label='w_p',hold=hold)
        pl.plot(ddw,label='ddw')
        pl.plot(dwsm,linewidth=3,label='sm(dw)')
        pl.plot(dw/10,label='dw_pdt/10')
        pl.scatter([wb, wbf, icent, wef, we],[0,200,300,250,275])
        pl.plot([wb,we],[0,0],label='b--e')
        pl.plot([wbf,wef],np.ones(2)*maxddw*wpmax/100,'o-',linewidth=2,label='bf-ef')
        pl.ylim(np.array([-1.1,1.1])*max(abs(dwsm)))
        pl.title(shot)
        pl.legend()
    debug_(max(pyfusion.DEBUG, debug),2, key='flat_top')
    #return(times[wb], times[we],(wb,we,wbf,wef,icent)) # used to ignore wbf,wef
    return(times[wbf], times[wef],(wb,we,wbf,wef,icent))


def get_delay(shot):
    if shot>=85000: 
        delay = 0.0
        print('get_basic_diagnostics - should fix with fetch')

    elif shot>=46455: delay = 0.2
    elif shot>=46357: delay = 0.4
    elif shot>=38067: delay = 0.1
    elif shot>=36185: delay = 0.3
    elif shot>=36142: delay = 0.1
    elif shot>=31169: delay = 0.3
    else: delay = 0.1
    if pyfusion.DEBUG>0: print('delay',delay)
    return(delay)


def get_basic_diagnostics(diags=None, file_info=file_info, shot=54196, times=None, delay=None, exception=False, debug=0):
    """ return a list of np.arrays of normally numeric values for the 
    times given, for the given shot.
    Will access server if env('IGETFILE') points to an exe, else accesses cache
    This is the first version to specifically allow for access through pyfusion.cfg
    There are two types of access:
       I/  single diag on its own timebase
       II/ the original multi diag on a given timebase (i.e. that from flucstrcs)
    Stage 1 puts the file_info into .cfg file just for I/ single diag access.
    Ideally the file_info for II/ sho;d be in .cfg also.
    For stage I/, we call it with a  file_info dict constructed on the spot
    as a dictionary with one just entry (for diags[0]).
    """

    global lhd_summary
    # if no exception given and we are not debugging
    # note - exception=None is a valid entry, meaning tolerate no exceptions
    # so the "default" we use is False
    if exception==False and debug==0: exception=Exception

    if diags is None: diags = "<n_e19>,b_0,i_p,w_p,dw_pdt,dw_pdt2".split(',')
    if len(np.shape(diags)) == 0: diags = [diags]
    if delay is None: delay = get_delay(shot)

    if times is None: 
        if len(diags)>1:
            times = np.linspace(0,4,4000)  # this is a crude guess.
        # else leave it None
    else:    
        # make sure it is an array
        times = np.array(times)

    

    vals = {}

    for diag in diags:
        if not(file_info.has_key(diag)):
            warn('diagnostic {0} not found in shot {1}'.format(diag, shot),stacklevel=2)
            vals.update({diag: np.nan + times})
        else:
            info = file_info[diag]
            varname = info['name']  # refers to name for igetfile - can contain ':' 
            subfolder = info['format'].split('@')[0]
            filepath = os.path.sep.join([localigetfilepath,subfolder,info['format']])
            if ':' in varname: (oper,varname) = varname.split(':')
            else: oper = None

            if info['format'].find('.csv') > 0:
                try:
                    test=lhd_summary.keys()
                except:    
                    csvfilename = acq_LHD+'/'+info['format']
                    if pyfusion.DEBUG>1: print('looking for lhd summary in' + csvfilename)
                    if not os.path.exists(csvfilename):
                        csvfilename += ".bz2"
                    print('reloading {0}'.format(csvfilename))
                    lhd_summary = read_csv_data(csvfilename, header=3)
                    # should make this more formal - last shots
                    # from an 'extra' file, and finally, from shot info
                if shot>117000: # fudge to get latest data
                    lhd_summary = np.load(acq_LHD+'/LHD_summary.npz')['LHD'].tolist()
                    print('loading newer shots from a separate file - fix-me')
                    #  val = lhd_summary[varname][shot-70000]    # not needed
                #  else:
                val = lhd_summary[varname][shot]    
                valarr = np.double(val)+(times*0)
            else:    
                # look for igetfile data in a few places
                debug_(max(pyfusion.DEBUG, debug), level=4, key='find_data')
                try:
                    dg = igetfile(filepath, shot=shot, debug=debug-1)
                except IOError:
                    try:
                        dg = igetfile(filepath+'.bz2', shot=shot, debug=debug-1)
                    except IOError:
                        try:
                            dg = igetfile(filepath + '.gz', shot=shot, debug=debug-1)
                        except exception, details:
                            if debug>0: print('diag at {fp} not found'
                                              .format(fp=filepath))
                            print(details)
                            dg=None
                            #break  # give up and try next diagnostic
                if dg is None:  # messy - break doesn't do what I want?
                    valarr=None
                else:
                    nd=dg.vardict['DimNo']
                    if nd != 1:
                        raise ValueError(
                            'Expecting a 1 D array in {0}, got {1}!'
                            .format(dg.filename, nd))

                    # pre re. w = np.where(np.array(dg.vardict['ValName'])==varname)[0]
                    matches = [re.match(varname,nam) 
                               != None for nam in dg.vardict['ValName']]
                    w = np.where(np.array(matches) != False)[0]
                    # get the column(s) of the array corresponding to the name
                    if (oper in 'sum,average,rms,max,min'.split(',')):
                        if oper=='sum': op = np.sum
                        elif oper=='average': op = np.average
                        elif oper=='min': op = np.min
                        elif oper=='std': op = np.std
                        else: raise ValueError('operator {o} in {n} not known to get_basic_diagnostics'
                                               .format(o=oper, n=info['name']))
                        valarr = op(dg.data[:,nd+w],1)
                    else:
                        if len(w) != 1:
                            raise LookupError(
                                'Need just one instance of variable {0} in {1}'
                                .format(varname, dg.filename))
                        if len(np.shape(dg.data))!=2:
                           raise LookupError(
                                'insufficient data for {0} in {1}'
                                .format(varname, dg.filename))
                             
                        valarr = dg.data[:,nd+w[0]]

                    tim =  dg.data[:,0] - delay

                    if oper == 'ddt':  # derivative operator
                        valarr = np.diff(valarr)/(np.average(np.diff(tim)))
                        tim = (tim[0:-1] + tim[1:])/2.0

                    if oper == 'ddt2':  # abd(ddw)*derivative operator
                        dw = np.diff(valarr)/(np.average(np.diff(tim)))
                        ddw = np.diff(dw)/(np.average(np.diff(tim)))
                        tim = tim[2:]
                        valarr = 4e-6 * dw[1:] * np.abs(ddw)

                    if (len(tim) < 10) or (np.std(tim)<0.1):
                        raise ValueError('Insufficient points or degenerate'
                                         'timebase data in {0}, {1}'
                                         .format(varname, dg.filename))

                    if times is not None:
                        debug_(max(pyfusion.DEBUG, debug), level=5, key='interp')
                        valarr = (stineman_interp(times, tim, valarr))
                        w = np.where(times > max(tim))
                        valarr[w] = np.nan
                    else:
                        times = tim
            if valarr is not None: vals.update({diag: valarr})
    # create an extra time array to allow a cross check
    vals.update({'check_tm':times})
    vals.update({'check_shot':np.zeros(len(times),dtype=np.int)+shot})

    return(vals)                

if not(os.path.exists(localigetfilepath)):
    os.makedirs(localigetfilepath)

get_basic_diagnostics.__doc__ += 'Some diagnostics are \n' + ', '.join(file_info.keys())


if __name__ == "__main__":
    global lhd_summary             # not sure if this does anything useful
    _var_default = """
numints=100
times=np.linspace(2,3,numints)
localigetfilepath = '/LINUX23/home/bdb112/datamining/cache/'

#shots=np.loadtxt('lhd_clk4.txt',dtype=type(1))
shots=[54194]
separate=0
diags="<n_e19>,b_0,i_p,w_p,dw_pdt,dw_pdt2,beta".split(',')
exception = IOError
verbose = 0
    """

    exec(_var_default)

    from pyfusion.utils import process_cmd_line_args
    exec(process_cmd_line_args())

    missing_shots = []
    good_shots =[]
    for shot in shots:
        try:
            basic_data=get_basic_diagnostics(diags,shot=shot,times=times)
            good_shots.append(shot)
        except exception:		
            missing_shots.append(shot)

    print("{0} missing shots out of {1}".format(len(missing_shots),(len(missing_shots)+len(good_shots))))

    if verbose>0: print('missing shots are {0}'.format(missing_shots))
    pl.plot(basic_data['check_tm'],basic_data['w_p'],hold=0)



"""
# I guess I copied this from LHD/read_igetfile to help with debugging this file.
#print(os.path.split(???__str__.split("'")[3])[0]+'/TS099000.dat')
filename='/home/bdb112/pyfusion/pyfusion/acquisition/LHD/TS090000.dat.bz2'
self=igetfile(filename)
dim=self.vardict['DimSize']
nv=self.vardict['ValNo']
nd=self.vardict['DimNo']
data3D=self.data.reshape(dim[0],dim[1],nd+nv)
for t in range(0,dim[0],5): pl.plot(np.average(data3D[t:t+10,:,4],0))
#for (t,dat) in enumerate(data3D[:,:]): print(dat[0])
tend=dim[0]/2
for t in range(0,tend,5): pl.plot(10*t+np.average(data3D[t:t+10,:,4],0),color=0.8*np.array([1,1,1]))

for t in range(0,tend,25): pl.plot(10*t+np.average(data3D[t:t+10,:,4],0),label=("%d ms" % (data3D[t,0,0])))

pl.legend()
"""
