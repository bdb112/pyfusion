# three colour phase plot for feature
# lenny 16  42/16 (3threads)
#       32  47/13.4 3 threads
import pylab as pl
import numpy as np
from pyfusion.utils.utils import warn
from pyfusion.data.DA_datamining import DA, report_mem
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden
from pyfusion.visual.window_manager import rwm, cm
from pyfusion.utils import compact_str
#from pyfusion.utils.dist import dist
from pyfusion.utils.dist_mp import dist_mp as dist
#from pyfusion.examples.density_cluster import dists
#run pyfusion/examples/density_cluster

def dists(cl_instance, instances):
    """ 14 sec cf 23 sec, and half the memory usage cf dists"""

    if cl_instance.dtype != np.float32: 
        cl_instance = np.array(cl_instance,dtype=np.float32)
    if instances.dtype != np.float32: 
        instances = np.array(instances,dtype=np.float32)
    if len(np.shape(instances))==1: 
        instances=np.reshape(instances,(1,len(instances)))
    return(dist(cl_instance, instances, squared=0, averaged=1))


_var_defaults="""
sel=None      # operates on phase data
subsel=None   # operates on cluster data
cl = 18
d_big = 0.5
d_sml = 0.1 # rms dist criterion for red points
d_med = 0.3 # rms dist criterion for green points
DAfilename = 'DA300_384_rms1_b5a_f16_diags.npz'
clusterfile = 'DA300_full_6000_70_bins_good.npz'
clearfigs = 1  # set to 0 to overlay (not great)
maxline=None  # fold suptitle lines to be about this length
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

try:
    da
    if oldDAfilename!=DAfilename: 
        1/0  # create an error to force reload
    
    print('Using old fs data')
except:
    print('loading {f}'.format(f=DAfilename))
    da=DA(DAfilename)
    oldDAfilename = DAfilename
    da.extract(locals(),'shot,phases,beta,freq,frlow,frhigh,t_mid,amp,a12')
    print('loading {f}'.format(f=clusterfile))

    if (sel is not None) and (subsel is not None) and len(subsel) != len(sel):
        raise ValueError('sel and subsel must have the same length')

try:
    cldata
    if oldclusterfile != clusterfile:
        1/0
    print('using previous clusterfile data' 
          ' - set oldclusterfile=None to prevent')
except:
    print('loading cluster data from {cf}'.format(cf=clusterfile))
    cldata = np.load(clusterfile)
    oldclusterfile = clusterfile
    oldsel = sel
    for k in cldata.keys(): exec("{v}=cldata['{k}']".format(v=k,k=k))
    if np.shape(sel) != np.shape(oldsel) or (sel != oldsel).any(): 
        warn('sel value tried to change from {o} to {n}'.
             format(o=oldsel, n=sel))
        sel = oldsel
    if subsel is not None:
        subset=subset[:,subsel]

start_mem = report_mem(msg='cluster_phases')
w5=np.where((dists(subset[clinds[cl][0]], phases[:,sel])<d_big) & (bw(freq,frlow,frhigh)) & (shot==shot))[0]; print(len(w5),len(unique(shot[w5])))
ph5=phases[w5]

wc=np.where(dists(subset[clinds[cl][0]], ph5[:,sel])<d_med)[0]
wcc=np.where(dists(subset[clinds[cl][0]], ph5[:,sel])<d_sml)[0]

sl_red = compact_str(np.unique(shot[w5[wcc]]))
sl_green = compact_str(np.unique(shot[w5[wc]]))
titl = 'red:d<{d_sml:.1g}:{slr}'.format(slr=sl_red, d_sml=d_sml)
suptitl = 'green:d<{d_med:.1g}:{slr}'.format(slr=sl_green, d_med=d_med)
if maxline is None:
    maxline = len(suptitl)**0.85
if len(suptitl)>maxline: 
    pieces = suptitl.split(',')
    suptitl = ''
    while len(pieces)>0:
        thislin = pieces.pop(0)
        while((len(thislin)<maxline) and len(pieces)>0):
            thislin += ','+pieces.pop(0)
        suptitl += thislin + '\n'

pl.figure(num='cl[{cl}] delta phase'.format(cl=cl))
if clearfigs: pl.clf()
for (i,ph) in enumerate(decimate(ph5,limit=1000)): pl.plot(ph[sel],'k',linewidth=.03,hold=i>0)
for (i,ph) in enumerate(decimate(ph5[wc],limit=1000)): pl.plot(ph[sel],'g',linewidth=.01,hold=i>-1)
for (i,ph) in enumerate(decimate(ph5[wcc],limit=1000)): pl.plot(ph[sel],'r',linewidth=.01,hold=i>-1)
pl.title(titl)
pl.suptitle(suptitl)
pl.show()
report_mem(start_mem)
pl.figure(num='cl[{cl}] freq-time'.format(cl=cl))
if clearfigs: pl.clf()
pl.plot(t_mid[w5],freq[w5],'.k')
pl.plot(t_mid[w5[wc]],freq[w5[wc]],'.g')
pl.plot(t_mid[w5[wcc]],freq[w5[wcc]],'.r')
pl.title(suptitl)

pl.figure(num='cl[{cl}] histo'.format(cl=cl))
if clearfigs: pl.clf()
pl.hist(dists(subset[clinds[cl][0]], ph5[:,sel]),50,log=True)
pl.title('{pc:.1g}% up to d_rms={d_big:.2g} rad'
         .format(pc=100*len(ph5)/float(len(phases)),
                 d_big=d_big))
pl.show()
