import subprocess
import sys
import pylab as pl
import numpy as np
from numpy import where
import pyfusion as pf
from glob import glob
import MDSplus as MDS  # TEMPORARY - SHOULD REALLY USE PYFUSION.

from pyfusion.data.DA_datamining import DA, report_mem, append_to_DA_file
from bdb_utils import process_cmd_line_args

def on(colls):
  for coll in colls:
    coll.set_visible(1)
  pl.show()

def off(colls):
  for coll in colls:
    coll.set_visible(0)
  pl.show()

def tog(colls):
  for coll in colls:
    coll.set_visible(not coll.get_visible())
  pl.show()


_var_defaults = """
#DA_file='DA_87206.npz'
sht = 86625
tf,tt=(.03, .04)
tf,tt=(.0, .06)
reread=True
thr=1.5
scl=500
clim=None
boydsdata=1
ylim=(0,100)
xlim=(0,0.06)

time_range=None
channel_number=0

"""
exec(_var_defaults)
exec(process_cmd_line_args())


dev_name="H1Local"
device = pf.getDevice(dev_name)
diag_name='H1ToroidalAxial' 
NFFT=2048 
noverlap=NFFT*7/8 
hold=1
shot_number = sht
cmap=cm.jet   # see also cm.gray_r etc

if DA_file is not None:
  try:
      da = DA(DA_file,load=1)
  except:
      print('DA_file {df} not found'.format(df=DA_file))

try:
    this_shot
except:
    this_shot=None

if not sht in da.da['shot']:
    reread=1
    print('shot {sht} not found, highest in {n} is {h}, rereading different shot'
          .format(sht=sht,n=da.name,h=np.max(da.da['shot'])))
pl.figure()
d = device.acq.getdata(shot_number, diag_name)
if time_range != None:
    dr = d.reduce_time(time_range)
else:
    dr = d
dr.subtract_mean().plot_spectrogram(noverlap=noverlap, NFFT=NFFT, channel_number=channel_number, hold=hold, cmap=cmap)

if ylim is not None:
  pl.ylim(ylim)

if xlim is not None:
  pl.xlim(xlim)

  # SHOULD REALLY GET THIS USING PYFUSION....
  h1tree = MDS.Tree('h1data', sht)
  main_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_main:I2').data()
  sec_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_sec:I2').data()
  kh=float(sec_current)/float(main_current)
  pl.title(pl.gca().get_title() + str(', k_h={kh:.2f}'.format(kh=kh)))
pl.show()

if reread:
    cmd = 'python pyfusion/examples/gen_fs_bands.py n_samples=None df=1e3  max_bands=3 dev_name=H1Local shot_range=[{sht}] diag_name=H1ToroidalAxial overlap=2.5 exception=Exception debug=0   n_samples=None seg_dt=0.0005 time_range=[{tf},{tt}] separate=1 info=2 method="rms"'.format(sht=sht,tf=tf,tt=tt)
    sub_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    print('waiting for prep_fs') 
    (resp,err) = sub_pipe.communicate()

    fname = 'PF2_150804{sht}s'.format(sht=sht)
    print(err)
    with open(fname,'w') as txtfile:
        txtfile.write(resp)
    this_shot = sht

    DA_file = 'DA_{sht}.npz'.format(sht=sht)
    mcmd = '''python pyfusion/examples/merge_text_pyfusion.py 'file_list=np.sort(glob("{fname}"))' exception=Exception save_filename={DA_file}'''.format(sht=sht,fname=fname[0:8]+'*s',DA_file=DA_file)
    sub_pipe = subprocess.Popen(mcmd,  shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    print('merging') 
    (resp,err) = sub_pipe.communicate()
    print(resp,err)

    da = DA(DA_file,load=1)

da.extract(locals())

if boydsdata:
    ph = -phases
else:
    ph = phases

w15=where((ml[15].one_rms(ph)<thr) & (shot==sht))[0];len(w15)
w4=where((ml[4].one_rms(ph)<thr) & (shot==sht))[0];len(w4)
w=np.union1d(w4,w15)
colls = []
if len(w)>0: colls.append(pl.scatter(t_mid[w],freq[w],scl*amp[w],c='b',label='n=5/m=4'))

w6=where((ml[6].one_rms(ph)<thr) & (shot==sht))[0];len(w6)
w1=where((ml[1].one_rms(ph)<thr) & (shot==sht))[0];len(w1)
w10=where((ml[10].one_rms(ph)<thr) & (shot==sht))[0];len(w10)
w=union1d(w1,w10)
if len(w)>0: colls.append(pl.scatter(t_mid[w],freq[w],scl*amp[w],c='g',label='n=4/m=3'))
pl.legend()

#%run -i pyfusion/examples/plot_specgram.py shot_number=sht dev_name=H1Local "diag_name='H1ToroidalAxial'" NFFT=2048 noverlap=NFFT*7/8 hold=1

if clim is not None:
    pl.clim(clim)
pl.show()
