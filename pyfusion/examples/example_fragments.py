import pyfusion as pf
#_PYFUSION_TEST_@@Skip
pf.config.get('global','database')  #  'sqlite:///sqlite.txt'
pf.read_config('shaun_feb_2010.cfg')

from pyfusion.conf.utils import get_config_as_dict
get_config_as_dict('Device','H1')

get_config_as_dict('Diagnostic','H1PoloidalAll')


import pyfusion as pf
h1=pf.getDevice("H1")
data=h1.acq.getdata(70071,'H1ToroidalAxial')
data.meta.keys()
data.plot_signals()

# overlay fs on spectrum
run examples/plot_specgram.py dev_name='LHD' shot_number=27233 hold=0 time_range=[.35,.5] NFFT=256 noverlap=220
run examples/plot_text_pyfusion.py filename='PF2_120229_MP_27233_27233_1_256.dat' hold=1 min_e=0.8 freq_scale=1e3
colorbar();xlim(0.35,.5);ylim(0,150000)

# mode identification overlaid on spectrum - used in LHD report Feb 2012
run examples/plot_specgram.py dev_name='LHD' shot_number=27233 hold=0 time_range=[.35,1.5] NFFT=256 noverlap=220 
clim(-210,-40)
run examples/mode_identify_example_2012.py hold=1 fsfile='PF2_120229_MP_27233_27233_1_256.dat'
xlim(0.35,.5);ylim(0,150000)
xlabel('Time (s)'); ylabel('Frequency (kHz)')

# chirp following example
run examples/plot_text_pyfusion.py filename='PF2_120229_MP_27233_27233_1_256.dat' hold=1 min_e=0.8 freq_scale=1e3 plot=1 time_range=[0.35,.4]

locals().update(extract_vars(ds[sind],['freq','t_mid']))
fs_id=sind  # a rather arbitrary ID number, but at least it can relate back to the file


; convert Shaun`s pickle to a dictionary of arrays.
sks = a.keys()  # serial_keys
ks = a[sks[0]].keys() # data_keys
# make a dictionary of arrays (instead of a dictionary of dictionaries)
d={}
for k in ks: d.update({k: [a[sk][k] for sk in sks]})
# extract into actual arrays - <
for k in ks: exec("{0}=array(d['{0}'])[:]".format(k))
freq=freq/1e3

#imshow(coil_1x_prop.T,extent=[0,1,0,300],aspect='auto',origin='lower',interpolation='nearest')

import pickle

pickle_file_name = '/home/bdb112/datamining/flucstrucs_pickle/16Jan-17-44-May19H1ToroidalAxial.txt.picklereduced'

a = pickle.load(open(pickle_file_name))

import pylab as pl
eps=1e-3
bins=(120,200)
extent = array([[0,1.2],[0,1e5]])

amp = [norm([coil_1x[i],coil_1y[i], coil_1z[i]]) for i in range(len(coil_1x))]
Hx,xed,yed=histogram2d(kh+eps,freq*sqrt(ne),bins=bins,weights=abs(coil_1x)/amp,range=extent)
Hy,xed,yed=histogram2d(kh+eps,freq*sqrt(ne),bins=bins,weights=abs(coil_1y)/amp,range=extent)
Hz,xed,yed=histogram2d(kh+eps,freq*sqrt(ne),bins=bins,weights=abs(coil_1z)/amp,range=extent)
# weights
Hw,xed,yed=histogram2d(kh+eps,freq*sqrt(ne),bins=bins,range=extent)

fig = pl.figure()
ax=fig.add_subplot(311)
CS=ax.imshow((Hx/Hw).T,aspect='auto',interpolation='nearest',origin='lower',extent=extent.flatten(),cmap=cm.gray_r)
fig.colorbar(CS,ax = ax)

ax2=fig.add_subplot(312,sharey=ax, sharex=ax)
CS=ax2.imshow((Hy/Hw).T,aspect='auto',interpolation='nearest',origin='lower',extent=extent.flatten(),cmap=cm.gray_r)
fig.colorbar(CS,ax = ax2)

ax3=fig.add_subplot(313,sharey=ax, sharex=ax)
CS=ax3.imshow((Hz/Hw).T,aspect='auto',interpolation='nearest',origin='lower',extent=extent.flatten(),cmap=cm.gray_r)
fig.colorbar(CS,ax = ax3)
