""" First attempt at real-time flucstruc processing 
Assume that the flucstruc files will be made remotely (by h1svr) or
locally, and that all files for that shot will be processed - e.g. if the preproessing is split over multiple processors.
"""
import subprocess
import sys
import pylab as pl
from matplotlib import cm
import numpy as np
from numpy import where
import pyfusion as pf
from glob import glob
from time import localtime
import MDSplus as MDS  # TEMPORARY - SHOULD REALLY USE PYFUSION.

from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file
from pyfusion.utils import process_cmd_line_args
from pyfusion.visual.sp import tog

_var_defaults = """
#DA_file='DA_87206.npz'
DA_file=""
sht = 86625
reprocess=False
thr=1.5
scl=500
clim=None
boydsdata=1 # this can also be used to check for random matches - should give 0 hits of flipped
ylim=(0,100)
xlim=(0,0.06)
remerge=0
time_range=[0.0,0.06]
channel_number=0

"""
exec(_var_defaults)
exec(process_cmd_line_args())

(tf,tt) = time_range
dev_name="H1Local"
device = pf.getDevice(dev_name)
diag_name='H1ToroidalAxial' 
NFFT=2048 
noverlap=NFFT*7/8 
hold=1
shot_number = sht
cmap=cm.jet   # see also cm.gray_r etc

tm=localtime()
hdr = str('PF2_{yy:02d}{mm:02d}{dd:02d}_'
          .format(yy=tm.tm_year-2000,mm=tm.tm_mon,dd=tm.tm_mday,hh=tm.tm_hour))

# a local da wins, if not there then try DA_file
try:
    da
except:
    print('No da, check if there is a file')
    if DA_file is not None:
        try:
            da = DA(DA_file,load=1)
        except:
            print('DA_file {df} not found'.format(df=DA_file))



flucfiles = '{hdr}*{sht}*'.format(sht=sht,hdr=hdr)
if not sht in da.da['shot']:
    print('shot {sht} not found, highest in {n} is {h}'
          .format(sht=sht,n=da.name,h=np.max(da.da['shot']))),
    # look for the flucfile remotely
    print(' acessing remote - may hang if sshfs')
    if len(glob('/h1svr2/tmp/'+flucfiles))>0:
        flucfiles = '/h1svr2/tmp/' + flucfiles
        remerge=1  # 
        print('using remote files')
    else:
        reprocess = 1
        print('will reprocess locally')

pl.figure()
d = device.acq.getdata(shot_number, diag_name)
if time_range != None:
    dr = d.reduce_time(time_range)
else:
    dr = d
dr.subtract_mean().plot_spectrogram(noverlap=noverlap, NFFT=NFFT, channel_number=channel_number, hold=hold, cmap=cmap)
ex = pl.gca().figbox.extents


if ylim is not None:
    pl.ylim(ylim)

if xlim is not None:
    pl.xlim(xlim)

if clim is not None:
    pl.clim(clim)

# SHOULD REALLY GET kappa USING PYFUSION....
h1tree = MDS.Tree('h1data', sht)
main_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_main:I2').data()
sec_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_sec:I2').data()
kh=float(sec_current)/float(main_current)
pl.title(pl.gca().get_title() + str(', k_h={kh:.2f}'.format(kh=kh)))

pl.show()

if reprocess:
    remerge=1
    cmd = 'python pyfusion/examples/gen_fs_bands.py n_samples=None df=1e3  max_bands=3 dev_name=H1Local shot_range=[{sht}] diag_name=H1ToroidalAxial overlap=2.5 exception=Exception debug=0   n_samples=None seg_dt=0.0005 time_range=[{tf},{tt}] separate=1 info=2 method="rms"'.format(sht=sht,tf=tf,tt=tt)
    sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    print('waiting for prep_fs') 
    (resp,err) = sub_pipe.communicate()

    # use a fixed (_) variant of the flucfiles wildcard use to save the processed flucs.
    print(err)
    with open(flucfiles.replace('*','_'),'w') as txtfile:
        txtfile.write(resp)

if remerge:  #  
    #tmpDA_file holds the merged flucstrucs until they can be merged into da
    tmpDA_file = '/tmp/DA_{sht}.npz'.format(sht=sht)
    # flucfiles is the wildcard name for the files processed on h1svr or locally
    mcmd = '''python pyfusion/examples/merge_text_pyfusion.py 'file_list=np.sort(glob("{tfile}"))' exception=Exception save_filename={DA_file}'''.format(sht=sht,tfile=flucfiles,DA_file=tmpDA_file)
    sub_pipe = subprocess.Popen(mcmd,  shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    print('merging') 
    (resp,err) = sub_pipe.communicate()
    print(resp,err)
    newda = DA(tmpDA_file,load=1)
    da.append(newda.da)
    
    print('{n} shots in {da}'.format(n=len(np.unique(da.da['shot'])),da=da.name))

da.extract(locals())

if boydsdata:
    ph = -phases
else:
    ph = phases

ws = np.where(sht == shot)[0]
ph = ph[ws]

print('where on {0:,}x{1} phases '.format(*np.shape(ph))),
w15=where((ml[15].one_rms(ph)<thr))[0];len(w15)
w4=where((ml[4].one_rms(ph)<thr))[0];len(w4)
w=np.union1d(w4,w15)
colls = []
if len(w)>0: colls.append(pl.scatter(t_mid[w],freq[w],scl*amp[w],c='b',label='n=5/m=4'))
print('{l} hits on 5/4'.format(l=len(w))),

w6=where((ml[6].one_rms(ph)<thr))[0];len(w6)
w1=where((ml[1].one_rms(ph)<thr))[0];len(w1)
w10=where((ml[10].one_rms(ph)<thr))[0];len(w10)
w=union1d(w1,w10)
if len(w)>0: colls.append(pl.scatter(t_mid[w],freq[w],scl*amp[w],c='g',label='n=4/m=3'))
print('{l} hits on 4/3'.format(l=len(w)))
pl.legend()
ax=pl.gca()
inset=0.15
pl.axes([ex[0]+0.05,ex[3]-inset-0.05,inset,inset],axisbg='y')
wa=np.where(a12[ws]<=1)[0]  # only needed because of a12>0 bug....
pl.hist(a12[ws[wa]],log=True)
pl.title('a12')
pl.sca(ax)
#%run -i pyfusion/examples/plot_specgram.py shot_number=sht dev_name=H1Local "diag_name='H1ToroidalAxial'" NFFT=2048 noverlap=NFFT*7/8 hold=1

pl.show()
