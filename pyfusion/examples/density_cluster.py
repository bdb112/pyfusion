""" Performance - with 18M x 14 phases CWGM dataset float32, with int indices, used 16GB to density cluster, and other ops took and additional 5GB swap.
At face value, this is 1k/entry, much higher than the typical 200-300 bytes.
Loading (copying dd, and extracting) the float32 version alone takes 5.6GB.
loading the float16 version alone (bare: no b_0)takes 3.3GB.
As these include 3 copies, space can be reduced to 40% by dd=0, DA300=0

30 secs for 20 million 
25ns
time for i in range(20*1000*1000): d.update({tuple(random.random(14).tolist()):i
New "unsafe" method takes 680 sec and 1.4GB for 16M of 18M phases
2.5GB after 0.85 Before - so  1.65GB for 16M entries
})
Idea: put each unique hash or the index itself into one of 1000-1M  an arrays until
one array is full.  Then empty all arrays that are more than half full into their dict 
_PYFUSION_TEST_@@sel="range(5)"@@PRE@from pyfusion.data.DA_datamining import da ; da=da(dd=False) ; da.extract(locals())
"""
import numpy as np
import pylab as pl
from numpy import float32, sqrt
from pyfusion.data.histogramHD import histogramHD

colorset=('b,g,r,c,m,y,k,orange,purple,lightgreen,gray'.split(',')) # to be rotated
from pyfusion.utils.utils import fix2pi_skips, modtwopi

def dist2(a,b, method='euler'):
    ax = (max(len(np.shape(a)),len(np.shape(b)))) - 1
    return(np.sum((modtwopi((np.array(a)- np.array(b)),offset=0)**2),ax))

def dists(a, instances, mask = None, method='euler'):
    """ calculate the RMS (not RSS) distance of the phase set to all phase sets
    based on std in new_mode_identify
    presently takes 1.4 secs for 1Mx14 float32.(100ns/elt
    4ns +,*, 21ns sqrt 43 ns modtwopi 26n (new fmod version)
    """
    if (len(a) != np.shape(instances)[-1]): 
        raise ValueError('shape of phases does not match shape of sample')
    if mask is None: mask=np.arange(len(a))

    if not(hasattr(instances, 'std')):
        print('make phase_array into an np array to speed up 100x')
        # the following assumes it is a structured array - it may be a list~!~
        instances = np.array(instances.tolist())  

    cc = np.tile(a[mask], (np.shape(instances)[0],1))
    # next line is a little faster
    # x=sum(modtwopi((phases[:,sel]-subset[clinds[cl][0]]),offset=0)**2,1)
    #
    # time x=sqrt(average(modtwopi((phases[0:1000000].astype(float32)+1)**2),1))
    # float32 should be faster than float or float16 (worst) but
    # because subset is float32, it is promoted to that precision automatically
    sq = (modtwopi(instances[:,mask]-a, offset=0))**2
    return(np.sqrt(np.average(sq,len(np.shape(instances))-1)))



def group_clusters(instances, radius=None):
    """ Assume the instances are in a priority order,
    return a list of lists which point to clustering, working from the 
    first (which is the most frequent down.
    """
    clusterinds = []
    dim = np.shape(instances)[1]
    if radius is None:
        radius = 0.1 * sqrt(dim)


    # start with all of them
    # remaining are the indices to the original list    
    remaining = range(len(instances))

    while len(remaining)>0:

        # leave the lead in the group tested - simplifies the logic.
        # find distance to all others.  The lead is always index [0]
        distances = [dist2(instances[remaining[0]], 
                           instances[ind])
                     for ind in remaining]
        # keep are indices to the current (shortened) list
        keep = np.where(np.array(distances) < radius**2)[0]
        # always one result (the lead), don't need to test
        if debug>2: print(len(remaining), distances)
        # work from the back so the elements are not moved til after
        # then reverse the result so the leader is first
        clusterinds.append(
            ([remaining.pop(i) for i in np.sort(keep)[::-1]])[::-1])
        # print(len(keep), keep)
    return(clusterinds)

def group_clustersold(instances, radius=None):
    """ Assume the instances are in a priority order 
    """
    clusters = []
    dim = np.shape(instances)[1]
    if radius is None:
        radius = 0.1 * sqrt(dim)

    lead = 0
    # start with all of them
    remaining = instances.tolist()
    while len(remaining)>0:
        if (remaining%100) == 0: print('{r} remaining'.format(r=remaining))
        # leave the lead in the group tested - simplifies the logic.
        # find distance to all others
        distances = [dist2(remaining[lead], instance) for instance in remaining]
        keep = np.where(np.array(distances) < radius**2)[0]
        # always one result (the lead), don't need to test
        clusters.append([remaining.pop(i) for i in np.sort(keep)[::-1]])
        # print(len(keep), keep)
    #if debug: 1/0    
    return(clusters)

_var_defaults="""

method='unsafe'
mincount=3
debug=0  # a numeric value controlling the amount of debugging info
hold=0
horiz=True  # horizontal legend for the first plot
n_bins = 48  # number of bins to create in each dimension (change requires re-run)      
sel = range(0,10)   # indices used to select probes out of the array (change requires re-run at present) 
sel.extend(range(12,16))
n_plot = 50  # number of clusters that will be plotted  - presently NOT properly immplemented
plot_unclustered=0 # plot all the bins, unclustered.
clmin=0    
clmax=9999 # clmin, clmax control what is plotted - nothing else
clgrey=0
dither=0.05  # dither in radians applied to phase
dphase = None  # cells closer than this will be made into a cluster
scale=0.01  # scale factor for width of lines
min_count = None  # this count is the smallest visible - None sets to the smallest of the n_plot selected
order_by_total=True # order clusters by their total polulation rather than by their largest bin
try:
    result
except:
    result=None  # will not be None after a clustering - so can re run without reclustering

"""
exec(_var_defaults)

from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if hold == 0 and (plot_unclustered) : pl.clf()

# this trick is to catch the name of the dataset if it is da
try:
    da
except:
    da = None


if result is None:
    hHD = histogramHD(phases[:,sel],bins = n_bins,method=method)
    # the nominal phase of a bin (probably the lower boundaries of a bin)
    phslist = hHD.d.keys()
    counts = array([hHD.d[k] for k in phslist])
    totcounts = np.sum(counts)
    w3 = np.where(counts>=mincount)[0]
    if len(w3)>1000:
        print('only processing where counts >= {mc}'.format(mc=mincount))
        phslist=[phslist[w] for w in w3]
        counts=counts[w3]

    proc_counts = np.sum(counts)
    inds = np.argsort(counts)
    result = (n_bins, mincount, sel)
    if method == 'safe':
        # this is a slow step (1 sec for 150k) - should include eps here too!
        phs = -np.pi+2*np.pi/n_bins*(0.5+np.array(phslist,dtype=float32))
    else:
        phs = []
        # there is probably a faster way than "ord" using numpy.char
        for phi in phslist:
            indices = [ord(c) for c in phi]
            # allow for bin centre
            phs.append(-np.pi+2*np.pi/n_bins
                        *(0.5+np.array(indices,dtype=float32)))
        phs = np.array(phs,dtype=float32)
    summ = str("{b} bins, \n processed {pc:,d}/{tc:,d}, mincount={mincount}"
               .format(b=n_bins, pc=proc_counts, tc=totcounts,
                       mincount=mincount))
    if da != None: summ = da.name + ': ' + summ

else:
    print('Using previous data')
    if result != (n_bins, mincount, sel): 
        old_result = result
        result = None
        raise LookupError("can't change n_bins sel or mincount without re-running completely\n (setting result=None), previous value in old_result")

print(summ)
if min_count is None: min_count = np.min(counts[inds[-n_plot:]])

dim = np.shape(phs)[1]

pl.rcParams['legend.fontsize']='small'

if plot_unclustered:
    for (i,ind) in enumerate(inds[-n_plot:][::-1]):   # reverse order
        label="{c}".format(c=counts[ind])
        if (i>200):
            if (i % 10)==0: label=label+'..'
            else: label = '_'+label

        pl.plot(phs[ind], label=label,
                linewidth=scale*n_bins*np.sqrt(np.max(counts[ind]+1-min_count,0)))

    if horiz:
        ncol = max(int(sqrt(1.5*n_plot)),12)
        pl.legend(ncol=ncol, mode="expand")
        pl.ylim(pl.ylim()[0],pl.ylim()[1]+n_plot*.03)
    else:
        pl.legend(ncol=3)
        pl.xlim(pl.xlim()[0],pl.xlim()[1]+5)

    pl.show()

pl.figure()

print('grouping')
#  subset is the nominal phases of the bins that made the cut.
subset = phs[inds[-n_plot:][::-1]]  # order so that highest is first
subset_counts = counts[inds[-n_plot:][::-1]]

if dphase is None:
    dphase = max(0.1*sqrt(dim),2 * 2*np.pi/n_bins)
clinds = np.array(group_clusters(subset,radius=dphase))

# order by total population
if order_by_total:
    pop = [np.sum(subset_counts[clind]) for clind in clinds]
    clinds = clinds[np.argsort(pop)][::-1]

# number in the smallest clinds bin
baseline = np.min([np.sum(subset_counts[clinds[i]]) 
                   for i in range(len(clinds))]) 
for (i,clind) in enumerate(clinds[clmin:clmax]):
    ii = i - clgrey  # anything less than 
    if ii<0: ii=0 
    linestyle=['-','--','-.',':',"-","--"][np.mod(int((ii/len(colorset))),6)]
    for (p,ind) in enumerate(clind):
        if p==0: label = str("{c}:{nc};{pop}"
                             .format(c=i,nc=len(clind),
                                     pop=np.sum(subset_counts[clind])))
        else: label = ''
        if dither != 0:
            dith = dither*(np.random.random(len(subset[ind]))-0.5)
        else: dith=0    
        colr = colorset[ii % len(colorset)]
        if i<=clgrey: colr = 'lightgray'
        pl.plot(subset[ind]+dith,color=colr, linestyle=linestyle,label=label,linewidth=scale*n_bins*(subset_counts[ind]-baseline*0.95)**0.5)
pl.title(summ+', plotted {tp:,d}'.format(tp=np.sum(subset_counts)))
if len(clinds) > 35:  # if too many to plot, switch to horizontal legend
    ncol = min(int(sqrt(1.5*(clmax-clmin))),9)
    pl.legend(ncol=ncol, mode="expand")
else:
    pl.legend()
pl.show()


"""
# cluster 14 centre of mass
sum(subset_counts[clinds[14]]* subset[clinds[14]].T,1)/sum(subset_counts[clinds[14]])

# look for the mode 2 out of the sml set in the full set -
# this reveals there is still an angle error of about .02 radians (60 nins)
# and 0.03 (50 bins) in the binning unbinning process?
# this is probably statisical, as only 5 bins (50) or 6 bins (60) were counted,
# so the average is notvery effective - only 1 bin wide in most dimensions.
 
from pyfusion.data.DA_datamining import DA
da=DA('DA300sml384_rms_5b_diags.npz')
time phases=da.da['phases']
%run -i pyfusion/examples/density_cluster.py n_bins=60 n_plot=400 scale=.01 method='unsafe' mincount=2
da18M=DA('DA300_384_rms1_b5a_f16_diags.npz')
time ph18=da18M.da['phases']
ph18=ph18[:,sel]
#Do a weighted average to get cluster centre
ref=sum(subset_counts[clinds[14]]* subset[clinds[14]].T,1)/sum(subset_counts[clinds[14]])
d=sum(abs(ref-ph18),1)
w=where(d<.32)[0];len(w)
plot d[w]

#To check on a particular cell
w3 is the first index (>=mincount)
then ins (orderby bin coint
then pop[::-1] order by cluster population
 

"""
