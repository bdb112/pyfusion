""" Script to merge a list of diagnostics and summary values into a data dictionary dd
Usage
dd=numpy.load('pyfusion/90091_MP2012_384_rms_1.npz')['dd'].tolist()
run -i pyfusion/examples/merge_in_basic_params.py
   optional arguments   
   exception=None   suppress no exceptions (for debugging)
   debug   set to 1 or higher to activate breakpoints
   diags   List of diagnostic names as defined in the info ditcionary (below is a shorthand)
              diags="<n_e19>,b_0,i_p,di_pdt,w_p,dw_pdt,dw_pdt2,beta,NBI".split(',')

Note: as of 14 Feb 2013, missing shot count is always []

===Earlier notes during development=== (obsolete now)
shot 46365, time 1.967 s

fsfile='54194.txt'
time run -i pyfusion/examples/plot_text_pyfusion.py
tim=ds['t_mid']
run -i pyfusion/examples/get_basic_params.py times=tim shots=[54194]
run -i pyfusion/examples/merge_in_basic_params.py

fsfile='/c/cygwin/home/bdb112/python/daves/pyfusion/MP512all.txt'
time run -i pyfusion/examples/plot_text_pyfusion.py
tim=ds['t_mid']
run -i pyfusion/examples/get_basic_params.py times=tim shots=[54194]
run -i pyfusion/examples/merge_in_basic_params.py
dd=load('fridd.npy').tolist()
run -i pyfusion/examples/test_lasso_fs.py
#save('fridd',dd)
for k in dd.keys(): exec("{0}=array(dd['{0}'])[:]".format(k))
w=where(beta>1)[0]
debug=0
sp(dd,'freq','beta','amp', size_scale=10000,hold=0, ind=w,col='t_mid',marker='s')
Y
sp(dd,'t_mid','freq','amp', size_scale=10000,hold=0, ind=w,col='beta',marker='s')
Y
colorbar()
ind=w
run -i pyfusion/examples/test_lasso_fs.py
=====


"""
import pyfusion
pyfusion.DEBUG=0

from pyfusion.data.DA_datamining import DA
#DAJ = DA('DA131128_HJ_ALL_50_52.npz',load=1)
#dd=DAJ.da


import os.path
import numpy as np
from pyfusion.acquisition.HeliotronJ.get_basic_diagnostics import get_basic_diagnostics, get_flat_top

debug=0
exception = IOError
exception = (IOError, LookupError, ValueError)

#dd={}
#for name in ds.dtype.names: dd.update({name: ds[name]})
diags="<n_e19>,b_0,i_p,di_pdt,w_p,dw_pdt,dw_pdt2,beta,NBI".split(',')
diags="b_0,w_p,dw_pdt,dw_pdt2,beta,NBI".split(',')
# diag _extra are the ones more likely to fail
diag_basic="IBTA,IBTB,IBHV,IBAV,IBIV,b_0".split(',')
diag_extra="n_e19b0,n_e19dL5,ech,NBI".split(',')
diag_extra="DIA135,MICRO01,NBIS3I,NBIS4I".split(',')
minshot=0
maxshot=999999 # higher than even LHD
shot_list = []
diags=diag_basic
diags_scalar="b_0,R_ax,Quad,Gamma".split(',')

import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())
# now to merge the two.
if len(np.shape(diags)) == 0: diags = [diags]

sz = len(dd['shot'])
missing_shots = []
good_shots =[]

ctr=0
 
if len(shot_list)==0:
    shots = np.unique(dd['shot'])
    wgt = np.where((shots >= minshot) & (shots <= maxshot))
    shot_list = shots[wgt]

for shot in shot_list:
    # ws is the set of indices corresponding to the shot
    ws = np.where(shot == dd['shot'])[0]
    if len(ws)==0:   # this is an impossible condition!
       raise LookupError('Impossible! could not find the expected shot {0}'.
                         format(shot))
    else: 
       try:
          times = dd['t_mid'][ws]
          basic_data = get_basic_diagnostics(diags,shot=shot,times=times,debug=1)
          if 'w_p' in diags:  # omit all this if not asked for w_p (to make sure we get Bo)
              (tstart,tend,inds) = get_flat_top(times=None, shot=shot) # None important
              flat_level = times*0
              w=np.where((times>tstart) & (times<tend))[0]
              flat_level[w] = 1.0
              if debug>0: print("len = {0}".format(len(w)))
              basic_data.update({'flat_level': flat_level})

          good_shots.append(shot)
       except exception as details:		
          missing_shots.append(shot)
          basic_data={}
          pyfusion.logging.warning("shot {s} not processed for diags, {info} {args}"
                                   .format(s=shot, info=details, args=details.args))

       if basic_data != {}:
           #bsign = np.sign(basic_data['b_0'][0])  # no need to correct so far
           # not used yet - maybe use for phases.
           for key in basic_data.keys():
               if debug>0: print(key)
               if key in dd:
                   ctr += 1
                   if np.mod(ctr,10) == 0: 
                       print('\nMerging in key {0} {1}'.format(key, shot)),
                   else:
                       print(key),
               else:    
                   print('Creating new key {0}'.format(key))               
                   dd.update({key: (np.zeros(sz)+np.nan).astype(pyfusion.prec_med)})

               #store it at the corresponding indices

               if key in ['not-yet']:#['w_p','i_p','dw_pdt','dw_pdt2','di_pdt']:
                   # this probably doesn't need astype (as it is a scatter)
                   dd[key][ws] = bsign*basic_data[key].astype(pyfusion.prec_med)
                       
               else:
                   dd[key][ws] = basic_data[key].astype(pyfusion.prec_med)
               if debug: print('{key}: avg={a:.3g}'.
                               format(key=key,a=np.average(dd[key][ws]))),

try:
    filename
except:
    filename='ddfile'
    print('filename defaulting to ', filename)

save_name = 'saved_'+os.path.splitext(os.path.split(filename)[1])[0]
print('Saving as {0}'.format(save_name))
#np.save(save_name,dd)

print("{0} missing shots out of {1}".format(len(missing_shots),(len(missing_shots)+len(good_shots))))

if verbose>0: print('missing shots are {0}'.format(missing_shots))

for key in diags:
        print('{0:10s}: {1:.1f}%'.format(key, 100.0*np.sum(dd[key]*0==0)/sz))
