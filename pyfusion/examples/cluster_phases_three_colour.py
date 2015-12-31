"""
 three colour phase plot for feature
 lenny 16  42/16 (3threads)
       32  47/13.4 3 threads

Use the cluster output from density clustering to cluster mode data
Presently works on the cluster lead (the most common element in the cluster)
To change cluster lead to the value calculated:
       subset[clinds[cl][0]][:]=cc5[0]
_PYFUSION_TEST_@@DAfilename='DA81115_sml.npz' clusterfile='ideal_toroidal_modes.npz' cl=6 "sel=arange(11,16)" "csel=range(0,5)"
'$DA/DA300_384_rms1_b5a_f16_diags.npz'

"""
import pylab as pl
import numpy as np
from pyfusion.utils.utils import warn
from pyfusion.data.DA_datamining import DA, report_mem
from pyfusion.data.convenience import between, bw, btw, decimate, his, broaden
from pyfusion.visual.window_manager import rwm, cm
from pyfusion.utils import compact_str, modtwopi
#from pyfusion.utils.dist import dist
from pyfusion.utils.dist_mp import dist_mp as dist
#from pyfusion.examples.density_cluster import dists
#run pyfusion/examples/density_cluster

import pyfusion
from pyfusion.debug_ import debug_

def dists(cl_instance, instances):
    """ 14 sec cf 23 sec, and half the memory usage cf dists"""

    if cl_instance.dtype != np.float32: 
        cl_instance = np.array(cl_instance,dtype=np.float32)
    if instances.dtype != np.float32: 
        instances = np.array(instances,dtype=np.float32)
    if len(np.shape(instances))==1: 
        instances=np.reshape(instances,(1,len(instances)))
    return(dist(cl_instance, instances, squared=0, averaged=1))

def ufn(fact,x,Nd):
    """ analytic form for the expected counts in a range of x in Nd dim space
    assuming uniform distribution
    """ 
    return(fact*x**(Nd-1))

def overlay_uniform_curve(hst, Nd, peak=1, background=0, colors=['r','g'], debug=0, **kwargs):
    """ plot a curve of the expected density of a unifrmly distributed
    population, matching at the "peak" and/or then background.  Assume
    variation is radius^(Nd-1) where Nd is the number of dimensions
    """
    # we assume that the mode (log or linear) is already set, so 
    # we just plot
    (cnt, x, patches) = hst
    x[1:] = (x[0:-1] + x[1:])/2   # mid point (except for first) 
    if np.isscalar(colors):
        colors = [colors, 'g']

    kwdm = dict(linewidth=2, linestyle='--') # for the main line
    kwdwh = dict(linewidth=2, linestyle='-')  # for the white underlay
    kwdm.update(kwargs)
    kwdwh.update(kwargs)

    xint = np.linspace(x[0],x[-1],300)  # interpolate x xaxis to 300 points
    lines = []
    if peak:
        ipeak = np.argmax(cnt)
        ifit = int(0.5 + ipeak*0.75) # aim to fit before the peak, but high enough that the count is reasonable
        fact = cnt[ifit]/ufn(1,x[ifit],Nd)
        pl.plot(xint, ufn(fact,xint,Nd),color = 'w', alpha=0.5, **kwdwh)
        lines.append(pl.plot(xint,  ufn(fact,xint,Nd),color = colors[0], **kwdm))

    if background:
        #ifit = -1
        fact = np.average(cnt[-4:]/ufn(1,x[-4:],Nd))
        pl.plot(xint, ufn(fact,xint,Nd),color = 'w', alpha=0.5, **kwdwh)
        lines.append(pl.plot(xint, ufn(fact,xint,Nd),color = colors[1], **kwdm))

    debug_((pyfusion.DEBUG,debug),1, key='overlay_uniform')    #if debug: 1/0
    background_counts = np.sum(ufn(fact, x, Nd))
    all_counts = np.sum(cnt)

    return(lines,all_counts - background_counts )

_var_defaults="""
sel=None      # operates on phase data
csel=None   # operates on cluster data  (previously 'subsel')
cl = 18
show_cc=True  # show cluster (not actually centroid - just first elt of cluster
show_dc=True # show data centroid
d_big = 0.5
d_sml = 0.1 # rms dist criterion for red points
d_med = 0.3 # rms dist criterion for green points
DAfilename = 'DA300_384_rms1_b5a_f16_diags.npz'
clusterfile = 'DA300_full_6000_70_bins_good.npz'
clearfigs = 1  # set to 0 to overlay (not great)
alpha = -.25  # -1 autoscales  alpha of black points
maxline=None  # fold suptitle lines to be about this length
# set uniform random_like this to test on uniformly distributed phases
uniform_random = '2*np.pi*(np.random.random([int(1e6), 16])-0.5)'
uniform_random = None
num_bins=50   # number of bins in the histogram
"""
exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if len(np.shape(cl)) == 0:
    cls = [cl]
else:
    cls = cl

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

    if (sel is not None) and (csel is not None) and len(csel) != len(sel):
        raise ValueError('sel and csel must have the same length')

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
        warn('sel value tried to change from {o} to {n} - i.e. cldata has different sel, '
             .format(o=oldsel, n=sel))
        sel = oldsel
    if csel is not None:
        subset=subset[:,csel]
# this contrivance allows us to test on uniformly distributed phases
if uniform_random is not None:
    print('evaluating {ur}'.format(ur=uniform_random))
    phases = eval(uniform_random)
    shot = np.array(np.shape(phases)[0]*[shot[0]])
    freq = np.array(np.shape(phases)[0]*[freq[0]])
    frlow = np.array(np.shape(phases)[0]*[0])
    frhigh = np.array(np.shape(phases)[0]*[9e9])
    t_mid = np.array(np.shape(phases)[0]*[t_mid[-1]])

cc5 = []  # cluster centres
for cl in cls:
    start_mem = report_mem(msg='cluster_phases')
# used to where all at once - but as the distance is most expensive, do
# shot and freq first
    w5_shot_freq = np.where((bw(freq,frlow,frhigh)) & (shot==shot))[0]; print(len(w5_shot_freq),len(np.unique(shot[w5_shot_freq])))
    # the [:,sel] below is to avoid gather ops on two indices at once.
    w5=np.where(dists(subset[clinds[cl][0]], phases[w5_shot_freq][:,sel])<d_big)[0]; 
    w5 = w5_shot_freq[w5] # refer back to the original array
# old "all at once" way
#    w5=np.where((dists(subset[clinds[cl][0]], phases[:,sel])<d_big) & (bw(freq,frlow,frhigh)) & (shot==shot))[0]; print(len(w5),len(np.unique(shot[w5])))
    print(len(w5),len(np.unique(shot[w5])))
    ph5=phases[w5]

    wc=np.where(dists(subset[clinds[cl][0]], ph5[:,sel])<d_med)[0]
    if len(wc)<1: raise ValueError('no points within avg radians of {dm} '.format(dm = d_med))
    wcc=np.where(dists(subset[clinds[cl][0]], ph5[:,sel])<d_sml)[0]

    sl_red = compact_str(np.unique(shot[w5[wcc]]))
    sl_green = compact_str(np.unique(shot[w5[wc]]))
    xlab='clind {cl}: {f}'.format(f=da.name, cl=cl)
    titl = 'red:d<{d_sml:.1g}: {slr}'.format(slr=sl_red, d_sml=d_sml)
    suptitl = 'green:d<{d_med:.1g}: {slr}'.format(slr=sl_green, d_med=d_med)
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
    if len(wcc)>0: 
        for (i,ph) in enumerate(decimate(ph5[wcc],limit=1000)): pl.plot(ph[sel],'r',linewidth=.01,hold=i>-1)
    if show_cc: pl.plot(subset[clinds[cl][0]],'y',linewidth=2,label='cluster ')
    # averaging will mess up if two pi jumps happen - average relative 
    # to assumed cluster centre and then shift back by that amount
    data_cent = (subset[clinds[cl][0]] 
                 + np.average(modtwopi(ph5[:,sel]-
                                       subset[clinds[cl][0]], 
                                       offset=0) ,0))
    if show_dc: pl.plot(data_cent,'--c',linewidth=2, label='data_centroid')
    cc5.append(np.average(ph5[:,sel],0))
    pl.legend()
    pl.title(titl)
    pl.suptitle(suptitl)
    pl.xlabel(xlab)
    pl.show(block=0)
    report_mem(start_mem)
    pl.figure(num='cl[{cl}] freq-time'.format(cl=cl))
    if clearfigs: pl.clf()
    if alpha < 0:
        alpha_ = (1.*len(w5))**(alpha)  # -0.25 10,000pts -> alpha_=.1
    else:
        alpha_ = alpha
    pl.plot(t_mid[w5],freq[w5],'.b', alpha=alpha_)
    pl.plot(t_mid[w5[wc]],freq[w5[wc]],'.g')
    pl.plot(t_mid[w5[wcc]],freq[w5[wcc]],'.r')
    pl.title(suptitl)
    pl.xlabel(xlab)

    pl.figure(num='cl[{cl}] histo'.format(cl=cl))
    if clearfigs: pl.clf()
    hst = pl.hist(dists(subset[clinds[cl][0]], ph5[:,sel]),num_bins,log=True)
    lines, excess = overlay_uniform_curve(hst, Nd = len(sel), 
                                          peak=1, background=1)
    pl.title('excess of {expc:.2g}% of {pc:.2g}% up to d_rms={d_big:.2g} rad'
             .format(pc=100*len(ph5)/float(len(phases)),
                     expc = 100*excess/float(len(phases)),
                     d_big=d_big))
    pl.ylabel('insts')
    pl.xlabel(xlab)
    if pl.isinteractive():
        pl.show(block=0)

pl.show(block=0)
