import numpy as np
import pylab as pl

# might be better to use   from mpl_toolkits.axes_grid1 import host_subplot

def plotdw(dd, w, hold=1, pad=5000, ax=None, debug=0):
    """ plot dw_pdt, freq on a fine scale near suspected events.
    the suspected events can be found with a query followed by find_sequence
    to group runs of instances. Alternatively  find_pulses will process 
    analog inputs in a similar way - both produce arrays of leading sample 
    and sample leangth.
    Can't see how to implement hold
    """
    if hold==0: print('hold not implemented yet')
    if ax is None: 
        ax=pl.gca()

    incs = np.diff(w)
    if (incs != 1).any(): warn('disjoint data at {0}'
                               .format(np.where(incs != 1)[0]))
    worig = w.copy()
    if (len(w) < pad):
        padeach = (pad-len(w))/2
        w = np.concatenate([np.arange(w[0]-padeach,w[0]),
                      w,np.arange(w[-1],w[-1]+padeach)])
    t = dd['t_mid'][w]
    ax.plot(t, dd['freq'][w],'g',label='f')
    dfdt = dd['dfdt'][w]
    wOK = np.where(np.isnan(dfdt)==False)[0]
    if len(wOK) < 5: print('probably too few non-nans in dfdt to plot')
    ax.plot(t, dd['dfdt'][w],'r',label='dfdt')

    ax2=ax.twinx()
    ax2.plot(t, dd['dw_pdt'][w],'b',label='dw_pdt')

    old_leg = ax.legend(loc=2)
    ax2.legend(loc=1)
    #ax2.add_artist(old_leg)
    
    ax.set_title('Shot {s}, N={n}'
                 .format(s=np.unique(dd['shot'][w]), n=np.unique(dd['N'][w])))
    # set the initial range to the original range given
    ax.set_xlim(min(dd['t_mid'][worig]), max(dd['t_mid'][worig]))
    if debug: 1/0
    pl.show()


def find_pulses(x, minlength=5, threshold=None, hysteresis=None, verbose=0):
    """like find sequence, but works with "analog" inputs (e.g. dw_p_dt)
    rather than logical.  Hysteresis is essential for analog inputs.
    Returns sample index of the start of each pulse and the lengths.
    """ 
    fig = pl.figure()
    ax = fig.add_subplot(111)
    x = np.array(x)
    (xmin, xmax)=(np.nanmin(x), np.nanmax(x))
    if threshold is None:
        threshhold = (xmin + xmax)/2
    if hysteresis is None:
        hysteresis = (xmax - xmin)/4

    onoffupper = (x > (threshhold + hysteresis/2)).astype(int)
    onofflower = (x < (threshhold - hysteresis/2)).astype(int)
    ons = np.where(np.diff(onoffupper) == 1)[0]
    offs= np.where(np.diff(onofflower) == 1)[0] # there will be a lot more offs
    # find an "on", then find the next off, and repeat from there
    # once the ons and offs are found, do further searches just on
    # the index arrays and they are shorter than the full arrays (e.g.deltas)
    ###pos = 0  # pos is an index to ons
    ##won = ons[pos]
    pstart = []
    plen = []
    pos=0  # pos is an index in the original array
    wwon = np.where(ons>pos)[0] # where is the next on rel to the ons array
    while(len(wwon)>0):
        won = ons[wwon[0]]          # won is similar to an intermediate pos
        wwoff = np.where(offs >= won)[0]
        if len(wwoff)==0:
            break
        else:
            pos = offs[wwoff[0]]
            pstart.append(won)
            plen.append(pos - won)
            wwon = np.where(ons >= pos)[0]  # could start at the current point
                                   # to save time, but need to check index
    return(np.array(pstart), np.array(plen))
    """
    #th_extra
    deltas=np.diff(onoff)
    ons = np.where(deltas == 1)[0]
    offs= np.where(deltas == -1)[0]
    
    # force to start with an on
    if ons[0]>offs[0]: offs=offs[1:]
    # and end with an off
    if ons[-1]>offs[-1]: ons=ons[0:-1]
    return(ons, offs-ons)
    """

def find_sequence(iarr, minlength=4, verbose=0):
    """ return an array of sequence starts and lengths for run 
    lengths >= minlength
    Returns sample index of the start of each run and the lengths.
    for example, [1,2,3,4,10,20,21,25,26,27,28,29] -> [0,6] and l=[4,5],
    because the pair 20,21 is too short.
    """
    """ SIMPLE MINDED
    for (i,j) in enumerate(iarr[1:]):
    if j-iarr[i]!=1: start=i
    else: 
    """
    # first eliminate runs of 1
    iarr = np.array(iarr)
    deltas = np.diff(iarr)
    w2 = np.where(deltas == 1)[0] # pointers to all who are followed by the next
    sstarts = []
    slens   = []
    in_seq = 0
    i = 0                         # i is going to step though the w2 array  
    start = w2[i]                 # start is the pointer to the start
    # the element of deltas pointed to by w2[0] is 1  (from the ==1 step)
    while (i<(len(w2)-1)):        # onlt w3 members are condidates
        while (deltas[i] == 1): 
            i += 1
            if i>= (len(w2)-1): break
            if verbose: print("{0} in seq".format(i))

        sstarts.append(start)
        slens.append(w2[i]-start)
        while (deltas[i] != 1): 
            i += 1
            if i>= (len(w2)): break
            if verbose: print("{0} not in seq".format(i))
        start = w2[i]

    return(sstarts, slens)
