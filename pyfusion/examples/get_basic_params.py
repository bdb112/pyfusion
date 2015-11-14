""" get the basic plasma params for a given shot and range of times
interpolation overhead only begins at 10k points, doubles time at 1Million!!

Replaced by device specific - e.g. acquisition/LHD/get_basic_diagnostics.py
At present, there is no example test routine for the above function - just run 
merge_basic_diagnostics.

"""

import pyfusion as pf
import pylab as pl
import numpy as np
import os
import pylab as pl
from pyfusion.debug_ import debug_
from pyfusion.acquisition.LHD.read_igetfile import igetfile

#from read_igetfile import igetfile
from  pyfusion.acquisition.LHD.read_igetfile import igetfile
from matplotlib.mlab import stineman_interp
from pyfusion.utils.read_csv_data import read_csv_data
from warnings import warn
import re

verbose = 0
numints=100
times=np.linspace(2,3,numints)
local_dir = '/LINUX23/home/bdb112/datamining/cache/'
this_file = os.path.abspath( __file__ )
this_dir = os.path.split(this_file)[0]
acq_LHD = this_dir+'/../acquisition/LHD/'

""" Make a list of diagnostics, and how to obtain them, as a dictionary of dictionaries:
Top level keys are the short names for the dignostics
Next level keys are 
   format:    format string to generate file name from shot.
   name: wild card name to find in the file header
 For the case of static parameters, the 'format' field is a csv file containing them all.
"""
file_info={}
#file_info.update({'n_e': {'format': 'fircall@{0}.dat','name':'ne_bar(3939)'}})
file_info.update({'<n_e19>': {'format': 'fircall@{0}.dat','name':'ne_bar\(3[0-9]+\)'}})
file_info.update({'i_p': {'format': 'ip@{0}.dat','name':'Ip$'}})
file_info.update({'di_pdt': {'format': 'ip@{0}.dat','name':'ddt:Ip$'}})
file_info.update({'w_p': {'format': 'wp@{0}.dat','name':'Wp$'}})
file_info.update({'dw_pdt': {'format': 'wp@{0}.dat','name':'ddt:Wp$'}})
file_info.update({'dw_pdt2': {'format': 'wp@{0}.dat','name':'ddt2:Wp$'}})
file_info.update({'beta': {'format': 'wp@{0}.dat','name':'<beta-dia>'}})
file_info.update({'NBI': {'format': 'nbi@{0}.dat','name':'NBI1(IAcc)'}})
file_info.update({'b_0': {'format': 'lhd_summary_data.csv','name':'MagneticField'}})
file_info.update({'R_ax': {'format': 'lhd_summary_data.csv','name':'MagneticAxis'}})
file_info.update({'Quad': {'format': 'lhd_summary_data.csv','name':'Quadruple'}})

global lhd_summary

def get_flat_top(shot=54196, times=None, smooth_dt = None, maxddw = None, hold=0, debug=0):
    if times is None: times=np.linspace(0.02,8,8020) ;  
    from pyfusion.data.signal_processing import smooth

    bp=get_basic_params(shot=shot,diags=['w_p','dw_pdt','b_0'],times=times)
    # assume get_basic corrects the sign
    w_p = bp['w_p']
    dw = bp['dw_pdt']
    w=np.where(w_p < 1e6)[0]
    len(w)
    cent = np.sum(w_p[w]*times[w])/np.sum(w_p[w])
    icent = np.where(times > cent)[0][0]
    print("centroid = {0:.1f}".format(cent))
    if maxddw is None: maxddw = 100
    if smooth_dt is None: smooth_dt = 0.1 # smooth for 0.05 sec
    dt = (times[1]-times[0])
    ns = int(smooth_dt/dt)
    smootharr = [ns,ns,ns]
    offs = len(smootharr*ns)
    dwsm = smooth(dw,n_smooth=smootharr)
    ddw = np.diff(dwsm)/dt
    # work away from the centroid until 2nd deriv exceeds maxddw
    # assume 100kJ /sec is ramp, and a chage of this over a second

    wb = int(0.5*offs) + np.nanargmax(dwsm)
    we = int(0.1*offs) + np.nanargmin(dwsm) # 
    wbf = offs + np.where(np.abs(ddw[0:icent])> maxddw)[0][-1]
    wef = offs + icent + np.where(np.abs(ddw[icent:])> maxddw)[0][0]
    if debug>0:
        pl.plot(w_p,label='w_p',hold=hold)
        pl.plot(dwsm,label='sm(dw)')
        pl.plot(ddw/10,label='ddw/10')
        pl.plot(dw,label='dw_pdt)')
        pl.scatter([wb, wbf, icent, wef, we],[0,500,1000,1500,2000])
        pl.plot([wb,we],[0,0],label='b--e')
        pl.ylim(array([-1.1,1.1])*max(abs(dwsm)))
        pl.title(shot)
        pl.legend()
    debug_(max(pyfusion.DEBUG, debug), key='flat_top')
    return(times[wb], times[we],(wb,we,wbf,wef,icent))


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
    print('delay',delay)
    return(delay)


def get_basic_params(diags=None, shot=54196, times=None, delay=None, debug=0):
    """ return a list of np.arrays of normally numeric values for the 
    times given, for the given shot.
    """

    global lhd_summary

    if diags is None: diags = "<n_e19>,b_0,i_p,w_p,dw_pdt,dw_pdt2".split(',')
    
    if delay is None: delay = get_delay(shot)

    if times is None: 
        times = np.linspace(0,4,4000)

    times = np.array(times)
    vals = {}
    # create an extra time array to allow a cross check
    vals.update({'check_tm':times})
    vals.update({'check_shot':np.zeros(len(times),dtype=np.int)+shot})
    for diag in diags:
        if not(file_info.has_key(diag)):
            warn('diagnostic {0} not found in shot {1}'.format(diag, shot),stacklevel=2)
            vals.update({diag: np.nan + times})
        else:
            info = file_info[diag]
            varname = info['name']
            if ':' in varname: (oper,varname) = varname.split(':')
            else: oper = None

            if info['format'].find('.csv') > 0:
                try:
                    test=lhd_summary.keys()
                except:    
                    print('reloading {0}'.format(info['format']))
                    lhd_summary = read_csv_data(acq_LHD+info['format'], header=3)

                val = lhd_summary[varname][shot]    
                valarr = np.double(val)+(times*0)
            else:    
                try:
                    dg = igetfile(local_dir + info['format'], shot=shot)
                except IOError:
                    try:
                        dg = igetfile(local_dir + info['format']+'.bz2', shot=shot)
                    except IOError:
                        try:
                            dg = igetfile(local_dir + info['format']+'.gz', shot=shot)
                        except exception:
                            #debug_(1)
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
                    if len(w) != 1:
                        raise LookupError(
                            'Need just one instance of variable {0} in {1}'.
                            format(varname, dg.filename))

                    # get the column of the array corresponding to the name
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

                    valarr = (stineman_interp(times, tim, valarr))
                    w = np.where(times > max(tim))
                    valarr[w] = np.nan

            if valarr != None: vals.update({diag: valarr})
    debug_(max(pyfusion.DEBUG, debug), level=5, key='interp')
    return(vals)                

get_basic_params.__doc__ += 'Some diagnostics are \n' + ', '.join(file_info.keys())


#shots=np.loadtxt('lhd_clk4.txt',dtype=type(1))
shots=[54194]
separate=0
diags="<n_e19>,b_0,i_p,w_p,dw_pdt,dw_pdt2,beta".split(',')
exception = IOError

import pyfusion.utils
exec(pf.utils.process_cmd_line_args())

missing_shots = []
good_shots =[]
for shot in shots:
    try:
        basic_data=get_basic_params(diags,shot=shot,times=times)
        good_shots.append(shot)
    except exception:		
        missing_shots.append(shot)

if len(missing_shots)>0:
    print("******************"),
print("{0} missing shots out of {1}".format(len(missing_shots),(len(missing_shots)+len(good_shots))))
if len(missing_shots) == len(shots):
    raise ValueError('no good shots')

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
