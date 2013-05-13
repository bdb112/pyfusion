""" These routines calculate df/dt for points in the same shot which have been
identified as belonging to a mode group
pseudocode:
   for all shots
       for all mode groups (N,M)
           link points
saving back into the original data set via the indx
v1: does 250k shots in ~ 1min.  uses maxd and dfdtunit, bool for islead
v2: islead is now the count of points linked, minlen, tidy output
"""
import pylab as pl
import numpy as np
from warnings import warn
# first, arrange a debug one way or another
try: 
    from pyfusion.debug_ import debug_
except:
    def debug_(debug,level=2,key=None,msg='',*args, **kwargs):
        if debug>level:
            print("attempt to debug key = {0}, msg = {1}" 
                  " need boyd's debug_.py to debug properly".format(key, msg))
debug=1
verbose=1

def plotdfdt(dd, inds, join=False, **kwargs):
    shots = np.unique(dd['shot'][inds])
    for shot in shots[0:min(50,len(shots))]:
        pl.figure()
        Ns = np.unique(dd['N'][np.where((dd['shot'] == shot)
                                        & (abs(dd['N'])< 10))[0]])
        for (i,N) in enumerate(Ns):
            sub_inds = np.where((dd['shot'] == shot) & ((dd['N'] == N)))[0]
            tord = np.argsort(dd['t_mid'][sub_inds])
            pl.plot(dd['t_mid'][sub_inds], dd['freq'][sub_inds],'os1234'[i],
                    hold=(N!=Ns[0]),label=str(N))
            if join:
                for id in np.unique(dd['lead'][sub_inds]):
                    # can omit the sub_inds in the following two lines
                    if id>=0:
                        line_indices = np.where(id == dd['lead'])[0]
                        if len(line_indices) > 1:
                            print("index {0}, {1} points"
                                  .format(id, len(line_indices)))
                            pl.plot(dd['t_mid'][line_indices],
                                    dd['freq'][line_indices],'k')
        pl.legend()
        pl.title('Shot {0}'.format(shot))

        
def process_chirps_m_shot(t_mid, freq, indx, maxd = 4, dfdtunit=1000, minlen=2, plot=0, debug=0    ):
    """
    >>> indx = np.array([1,2,3,4]); freq = np.array([5,4,2,1])
    >>> t_mid = np.array([1,2,3,10])
    >>> process_chirps_m_shot( t_mid, freq, indx, maxd=4, dfdtunit=1, debug=0)
    3 connected of 4 chirping,
    (array([3, 0, 0, 0]), array([ 1,  1,  1, -1]), array([ -1.,  -2.,  nan,  nan], dtype=float32))
    """
    def ftdist(i_from, i_to, tm, fr, dfdtunit=dfdtunit):
        return(np.sqrt((fr[i_to]-fr[i_from])**2 + 
                       ((tm[i_to]-tm[i_from])*dfdtunit)**2))

    lines = [] # list of lists of indices for points in lines
    line = [0] # start at the first point
    if debug>3: print("processed {0},".format(len(freq))),
    for i in range(1,len(freq)):
        # if the current point is still coincident with the start, skip
        # else if it is also within maxd in ft space, append it.
        # this method has the flaw that two points at the same time 
        # terminate the line - but that should not happen.
        if t_mid[i]!=t_mid[line[-1]] and ftdist(line[-1],i,t_mid,freq) < maxd:
            line.append(i)
        else:
            if len(line) > 1: lines.append(line) # save it if len > 1
            line = [i]  # start a new one

    for line in lines:
        if plot: pl.plot(t_mid[line],freq[line],'-o')

        # if there are at least 3 points, and the df dt agrees with neighbours
        #  within 1kHz/ms, set df/dt - otherwise set to 0 (this line not impl?)
        # also should record the id of the line (-1 for no line) the points 
        # belong to, a good choice is the index (rel to the original 
        # dataset) of the first point.

    num = len(indx)
    lead = indx*0 - 1
    islead = np.zeros(num, dtype=int)
    dfdt = np.zeros(num, dtype=np.float32)
    dfdt[:]=np.nan
    chirping = 0
    for line in lines:
        if len(line) >= minlen:
            islead[line[0]] = len(line)  # marks this as the lead point
            chirping += len(line)
            # marks the points as belonging together
            lead[line] = indx[line[0]]   
            # only distinct dfdts are recorded - the last 
            # point is left without a dfdt
            dfdt[line[0:-1]] = ((freq[line[1:]] - freq[line[0:-1]])/
                                (t_mid[line[1:]] - t_mid[line[0:-1]])/
                                dfdtunit).astype(np.float32)
    print("{0} connected of {1} chirping,".format(chirping, len(freq))),
    debug_(debug, 4, key='detail')
    return(islead, lead, dfdt)

def process_chirps(dd, shots=None, Ns=None, Ms=None, maxd = 4, dfdtunit=1000, minlen=2,
                   plot=0, hold=0, clear_dfdt=False, debug=0, verbose=0):
    """ M and N default to all values present
    >>> 
    """
    # add islead, lead, dfdt entries
    if plot>0 and hold==0: pl.clf()  # plot(hold=0) runs slower here
    num = len(dd['shot'])
    if not dd.has_key('indx'): raise LookupError('dd needs and indx!')
    indx = dd['indx']
    if clear_dfdt or not dd.has_key('dfdt'):
        print('******* initialising dfdt data ************')
        dd.update({'islead': np.zeros(num, dtype=int)})
        dd.update({'lead': np.zeros(num, dtype=int)})
        dd['lead'][:]= -1
        dd.update({'dfdt': np.zeros(num, dtype=np.float32)})
        dd['dfdt'][:]=np.nan
    # extract the shot, M and N

    warn("need to choose a 'nan' equivalent in process_chirps")
    for k in ['shot', 'M', 'N']: 

        w = np.where(dd[k]!= -4)[0]  # right now it is -4 - but effect on colorscale?
        exec("all_{0}=np.array(dd['{0}'])[w]".format(k))

    if Ms == None: 
        Ms = np.unique(all_M) 
        warn('defaulting Ms to {0}'.format(Ms))
    if Ns == None: 
        Ns = np.unique(all_N) 
        warn('defaulting Ns to {0}'.format(Ns))
    debug_(debug,2,key='start_chirps')
    for shot_ in shots:    
        if verbose>0: print("\n{0}:".format(shot_)),
        for N_ in Ns:
            if verbose>0: print(" {0}:".format(N_)),
            for M_ in Ms:
                inds = np.where((shot_ == all_shot) & 
                             (M_ == all_M) & (N_ == all_N))[0]
    # extract, but order in time
                tord=np.argsort(dd['t_mid'][inds])
                if verbose > 4:
                    print("shot {0}, len tord={1}".format(shot_,len(tord)))
                for k in dd.keys(): 
                    if not hasattr(dd[k],'keys'):
                        exec("{0}=np.array(dd['{0}'])[inds[tord]]".format(k))
                (islead, lead, dfdt) = \
                    process_chirps_m_shot(t_mid, freq, indx, maxd=maxd, 
                                          dfdtunit=dfdtunit, minlen=minlen,
                                          plot=plot, debug=debug)
    # write them back into the dict dd
                if len(islead) != 0:            
                    if verbose > 2: print("indx={0}".format(indx))
                    dd['islead'][indx] = islead
                    dd['lead'][indx] = lead
                    dd['dfdt'][indx] = dfdt
                debug_(debug, 3, key='write_back')
    print("minlen = {m}, maxd = {d}, dfdtunit={df}".
          format(m=minlen, d=maxd, df=dfdtunit))
    if plot>0: pl.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

"""
run examples/mode_identify_example_2012.py hold=1 fsfile='PF2_120229_MP_27233_27233_1_256.dat'
run examples/mode_identify_example_2012.py hold=1 fsfile=/c/cygwin/home/bdb112/python/daves/pyfusion/MP512all.txt skip=4
#  create mode numbers
dd={}
# make into a dict
for name in ds.dtype.names: dd.update({name: ds[name]})
# add an index - only makes sense if the ds var is 1:1 with a file
# (no previous selection) - useful for storing info back into full dataset
dd.update({'indx':range(len(dd[dd.keys()[0]]))})
num = len(dd['shot'])
dd['N']=np.zeros(num, dtype=np.int16)
dd['N'][:]=-999
dd['M']=np.zeros(num, dtype=np.int16)
dd['M'][:]=-999
dd['N'][neq2]=2
dd['N'][neq1]=1
np.dd['N'][neq0]=0


"""
