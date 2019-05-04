""" Check that the new analysis (for the params hard codede here) performs as well as the old in a list of LP DA files
run compare_analysis.py <filename>
run compare_analysis.py <filename> tmax   # will analyse only for tmax in the middle of the given data
run compare_analysis.py @file_containing_list_of_names
  no arguments assumes a list of name in variable files
A single file (list of length 1) will generate an ediff of the info attributes

hard coded to do a re-run of LP0107 with hard_coded params and compare the results with those channels of the given file.

_PYFUSION_TEST_@@LP/LP20160309_52_L53_2k2small.npz 0.008
"""
import numpy as np
import pyfusion
import os, sys
import matplotlib.pyplot as plt
from pyfusion.data.DA_datamining import Masked_DA, DA
from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup

files = ['LP/LP20160309_52_L53_2k2small.npz']

if len(sys.argv)>1:
    arg1 = sys.argv[1] 
    if arg1.startswith('@'):
        files = np.loadtxt(arg1[1:], dtype='S')
    else:
        files = [arg1]
else:
    print('using existing list <files> if run with -i')

tmax = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
for n,f in enumerate(files):  # (bads):
    da = DA(f)
    shot = [da['date'][0], da['progId'][0]]
    diag = 'W7X_L53_LP0107' if '53' in da['info']['channels'][0] else 'W7X_L57_LP0107'
    LP = Langmuir_data(shot, diag, 'W7X_L5UALL')
    kwargs = dict(threshchan=0,initial_TeVfI0= {'I0': None, 'Te': 30, 'Vp': 5}, fit_params=(dict(alg='amoeba', lpf=21)), dtseg=2000, overlap=2)
    old_t_range = da.infodict['params']['t_range']
    if old_t_range is None:
        old_t_range = np.array(da.infodict['params']['t_comp']) + 0.2  # guess
    if tmax is None:
        t_range = old_t_range
    else:
        t_range = np.mean(old_t_range) + np.array([-tmax, tmax])/2
    kwargs.update(dict(t_range=t_range))
    LP.process_swept_Langmuir(**kwargs)
    fname = '/tmp/{d}_{s}'.format(d=shot[0], s=shot[1])
    LP.write_DA(fname)
    cda = DA(fname)
    dach = da['info']['channels'] 
    cdach = cda['info']['channels']
    plt.figure(num='{f}'.format(f=f,n=n).replace(os.getcwd(),'.'))
    nrows = len(cdach)
    #  tchan refers to the smaller set just calculated
    for tchan in range(nrows):
        chan = None
        for i, ch in enumerate(dach):
            if dach[i] == cdach[tchan]:
                chan = i
        if chan is None:
            print("can't match {dach} with {cdach}"
                  .format(dach=dach, cdach=cdach))
            continue
        plt.plot(da['t_mid'][0:100], da['Te'][0:100,chan],'r')
        plt.plot(cda['t_mid'], cda['Te'][:,tchan], 'g',label='latest ' + dach[chan])
    plt.title('rest_swp {rs}, diag={d}'.format(rs=da['info']['params']['rest_swp'], d=diag))
    plt.legend(loc='best')
    plt.show(block=0)

if len(files) == 1:
    import tempfile, pprint
    names = []
    for _da in [da, cda]:
        fp, fname = tempfile.mkstemp()
        os.write(fp, _da.name + '\n')
        os.write(fp, pprint.pformat(_da['info']))
        os.close(fp)
        names.append(fname)

    if tmax is not None and tmax >= 0.01:  # very short tmax assumes we are testing - don't emacs
        os.system('ediff {f1} {f2}'.format(f1=names[0], f2=names[1]))
# leave the files there in tmp    
"""
!locate pyfusion/LP|grep LP20160309|grep to_|grep npz > /tmp/checkfiles
# to save the images that I am not happy with.
bads=[]
med=[]
bads.append(gcf().get_label()) ; fig.savefig(gcf().get_label().replace('npz','png'))
med.append(gcf().get_label()) ; fig.savefig(gcf().get_label().replace('npz','png'))
...etc
import json
json.dump(dict(bads=bads,med=med),open('compare_analysis_jan_31_2017','w'))
"""
