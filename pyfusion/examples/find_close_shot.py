
import numpy as np
import pylab as pl
 
def nansort(arr):
    return(np.sort(arr[np.where(np.invert(isnan(arr)))]))

def nanabs(arr):
    return(np.abs(arr[np.where(np.invert(isnan(arr)))]))

def nanargsort(arr):
    return(np.argsort(arr[np.where(np.invert(isnan(arr)))]))


def find_close_shot(sh, diags, limit=10, summarydb = LHD, subrange=None, debug=0, verbose=1):
    """ find the 'distance' to other shots in the subrange, and return the
    closest <limit> shots and the measure of closeness 
    diags is a , delimited list 
    optional : qualifies the type of test  
       beta:<1 means beta <1 (rahter than compareed to the template
       beta:0.1 means the difference between beta and template beta 
               is scaled by 0.1  (default is scaled by the average)

    subrange is the range of shots to restrict comparisons to
    """

    SDB = summarydb
    shots = SDB['Shot']  # boyd's notation - allows for cross check

    if subrange != None:
        inds = np.array(subrange)
        #inds = []  # this option too slow
        #for s in subrange:
        #    inds.append(where(s == shots)[0][0])
        #inds=np.array(inds)
    else:
        inds = shots  # inds is always a shot number as LHD indexed by shot
        raise ValueError('inds=shots doesn`t work - must use subrange=')
    # check
    wneq = np.where(LHD['Shot'][inds] != inds)[0]
    if len(wneq) > 0: print('Subrange shots {s} not in csv data, removing'.
                            format(s = inds[wneq]))
    weq = np.where(LHD['Shot'][inds] == inds)[0]
    inds = inds[weq]

    dist = []
    dgs = diags.split(',')
    
    keys = []
    for d in dgs:
        (op, rest) = ('','')  # initialise to nothing
        k = d.split(':')[0]
        keys.append(k)
        if len(d.split(':'))>1:
            rest = d.split(':')[1]
            if rest[0] in '><=': 
                rest = rest[1:]
        if op != '' : raise ValueError('no operators supported yet')
        if rest != '': 
            try:
                scl = eval(rest)
            except Exception as details:
                raise ValueError('value error in {r} parsing {d} {a}'
                                 .format(r=rest, d=details, a=details.args))
            if scl == 0: scl=1e-9
        else: scl =  np.average(nanabs(SDB[k]))
        dist.append((SDB[k][inds] - SDB[k][sh])/scl)

    d2 = np.sum((np.array(dist))**2,0)
    print(d2[np.where(inds==sh)[0]])
    w = np.where(d2 <9e99)[0]
    (uniq, index, inv) = np.unique(d2[w],return_inverse=True,return_index=True)
    nmatches = len(np.where(inv == 0)[0])
    if nmatches > limit:
        print('more matches {m} than limit'.format(m=nmatches))
    if debug: 1/0
    imax = min(limit,len(index))
    if verbose:
        pkeys = keys
        pkeys.insert(0,'Shot')
        print(''.join(['{k:^14}'.format(k=k) for k in pkeys]))
        pinds = np.insert(inds[index[0:imax]],0,sh)
        fmts = len(pkeys)*['{v:^14.3f}']
        fmts[0] = '{v:^14s}'
        for i in pinds:
            print(''.join([fmts[ii]
                           .format(v=LHD[k][i]) for (ii,k) in enumerate(pkeys)]))
            #print(''.join(['{v:14.3f}'
            #              .format(v=LHD[k][pinds[i]]) for k in pkeys]))
    return(inds[index[0:imax]],d2[index[0:imax]])


