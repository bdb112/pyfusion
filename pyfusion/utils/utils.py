import numpy as np

# this changes the behaviour of warnings
import warnings
import pyfusion

# this could run slower in python2 (used to be xrange)
def find_last_shot(fun, range=[1, 200000], ex=True, quiet=False):
    """ find the last shot for which fun(shot) does not return an error
        Actually more general than this.  Only tested for low lim yes, uplim no
        
    >>> def myfun(shot): return(shot <= 12345)
    >>> find_last_shot(myfun, quiet=True)
    12345
    """

    def mid(low, high):
        return(int((low+high)/2))

    def quiet_call(fun, arg, ex=True):
        try:
            res = fun(arg)
        except Exception:
            if (ex):  # if we are watching for exceptions, return False if we get one
                res = False
        if res is False and (not quiet): print('.'),
        return(res)
        

    maxits = 1000 # safety valve

    low = min(range)
    high = max(range)
    if not quiet_call(fun, low): 
        raise ValueError('low limit of {l} not there'.format(l=low)) 

    if quiet_call(fun, high): 
        raise ValueError('upper limit of {u} is there'.format(u=high)) 

    shot = mid(low, high)
    while (1):
        maxits -= 1
        if maxits < 0: raise Exception('infinite bisection?')
        res = quiet_call(fun, shot)
        # if not false, try higher
        if res is not False:
            low = shot
        else:
             high = shot   
        nextone = mid(low, high)
        #  print(maxits, nextone,high,low)
        if low == high-1:
            if (shot == high):
                if res: 
                    return(shot)
                else:
                    return(low)
            else: # shot must be low, and it fun must be true
                return(shot)

        shot = nextone


def compact_str(shot_numbers, max_changing_digits=2, min_run=4,  debug=False):
    """ returns a compact string represetation of shots.  e.g.
    [100,101,102,103,109,110,111,120] ==> "100..103,109,110,120"
    >>> compact_str([10000,10001,10002,10003,10009,10010,10101,10200,11000])
    '10000,1,2,3,9,10 10101 10200 11000'
    >>> compact_str(range(10010,10017),debug=False)
    '10010-10016'
    """
    def longest_sequence_length(iarr):
        if len(iarr)<2: return len(iarr)
        diffs=np.diff(iarr)
        other_size_steps = (diffs!=1).nonzero()[0]
        if len(other_size_steps) == 0: 
            if debug: print('other size steps', other_size_steps)
            return(len(iarr))
        else: 
            return other_size_steps[0]+1

    def leading_digits_in_common(s1,s2):
        if debug: print ('leading digits in common', s1, s2)
        if len(s1) != len(s2): return(0)
        for (i,c) in enumerate(s1):
            if s1[i] != s2[i]:
                #print("i=%d = %s" % (i, s1[i]))
                return(i)
        return(len(s1))

    # first look for a simple sequence
    shots=list(shot_numbers)  # to be sure we don't wipe out the original list
    sstr=''
    last_full_shot_str=''
    if debug: print ('compactstr')
    while len(shots)>0:
        ls=longest_sequence_length(shots)
        if ls>min_run:  # write a range specifically as from-to, leading comma removed at end
            sstr +=(',%d-%d') % (shots[0],shots[ls-1])
            last_full_shot_str = str(shots[ls-2])
            shots=shots[ls:]
            if debug: print('6','lfs',last_full_shot_str, shots)
        else: #elif ls==1:
            # somewhere here, could look for the case where putting in a new full shot
            # would help - i.e. if the digits are changing more slowly
            if debug: print ('else','lds',last_full_shot_str, shots)
            shstr=str(shots[0])
            ldic = leading_digits_in_common(shstr, last_full_shot_str)
            if ldic>=(len(shstr)-max_changing_digits) and len(shstr) == len(last_full_shot_str):
                sstr += ','+ shstr[ldic:]
                # don't update last full here, as it is not full!
            else:
                sstr += ' ' + shstr
                last_full_shot_str = str(shots[0])

            shots=shots[1:]

    if (sstr != '') and (sstr[0]) in ' ,': sstr = sstr[1:]
    return(sstr)
    #return str(shot_numbers)   # not yet implemented!


def get_local_shot_numbers(partial_name=None, verbose=0, local_path=None, 
                           number_posn=[0,5]):
    """ get shots present in path py extracting numbers from matching names
    This applies to local copies in .npz form.
    Defaults to local_path=pyfusion.config.get('global','localdatapath')
    partial name defaults to _MP1
    """ 
    from os import walk
    if local_path is None: 
        local_path = pyfusion.config.get('global','localdatapath')
    if partial_name is None: partial_name="_MP1"
    shotlist=[]
    for root, dirs, files in walk(local_path):
        for f in files:
            if f.find(partial_name)>=0: 
                if verbose>0: print (root,dirs,f)
                if root==local_path: shotlist.append(
                    f[number_posn[0]:number_posn[1]])
# the old way assumed a fixed position for the shot number
# this format will just get the number part but not if the number leads!
#                    niceify_name(f,format="%.0s%d"))
    try:
        return(np.array(shotlist,dtype=int))
    except:
        raise ValueError('shotlist contains non integers e.g.{s}'
                         .format(s=shotlist[0]))

# suppress this change altogether with colors=None in globals
# can't be changed at runtime.
if pyfusion.COLORS != None:
    orig_show = warnings.showwarning
    try: 
        from IPython.utils import coloransi
        tc=coloransi.TermColors()
        red = tc.LightRed
        normal = tc.Normal
    except None:
        (red, normal) = ('***','***')  # make error prominent if no clours

    def my_show(message, category, filename, lineno, file=None, line=None):
        orig_show(red + message.message + normal, 
                  category, filename, lineno, file, line)

    warnings.showwarning = my_show    
    if pyfusion.VERBOSE>0:
        print('warning will be shown' + red + ' like this' + normal)
# end of color warnings block

def warn(warning, category=UserWarning ,stacklevel=2, exception=None):
    """ Similar to warnings.warn, but includes info about the exception.
    e.g.  warn('extracting data from shot %d' % (shot), exception=ex)
    will print Exception type .., <args>: extraction data from shot 1

    Downside compared to print() is that it generates 2-3 lines
    (including line number of caller) instead of one.  Perhaps use for 
    infrequent or more dangerous situations
    - for "nagging" purposes, print is probably better.
    Perhaps this should always be used for exception catching.
    
    A minor downside to explicit detail in the message is that 
    multiple messages will be printed if the warning string changes 
    from one error to the next (unless warnings are disabled.)
    """
    if exception is None: exmsg=''
    else: exmsg = 'Exception "%s, %s": ' %  (type(exception),exception)
    # need the +1 so that the caller's line is printed, not this line.
    # also break the line where the 'red' colour starts
    warnings.warn('\n'+exmsg+warning, category, stacklevel=stacklevel+1)


def modtwopi(x, offset=np.pi):
    """ return an angle in the range of offset +-pi
    >>> print("{0:.3f}".format(modtwopi( 7),offset=3.14))
    0.717
    >>> print("{0:.3f}".format(modtwopi( -5.12),offset=0))
    1.163

    This simple strategy works when the number is near zero +- 2Npi,
    which is true for calculating the deviation from the cluster centre.
    does not attempt to make jumps small (use fix2pi_skips for that)
    but by the correct choice of offset, jumps can be reduced
    The difference between this and fix2piskips is that this
    routine always returns within a range of 2pi (around offset).
    """
    if not hasattr(x,'std'):
        x = np.array(x)

    """ fmod implementation - remainder is what we really want    
    #faster than %
    mininp = np.min(x)
    # remainder is fastest, (18ns cf 26ns (fmod) cf 43 (%) for float32) 
    # 14 May 2013 can't repeat this! all are about 18-19ns
    # fmod needs more work as it has different behaviour below 0
    # note also c version of remainder does it -1/2 to +1/2
    if (mininp-offset)<0: 
        add_pos =  2*np.pi * (1 + int((-(mininp-offset))/(2*np.pi)))
    else:
        add_pos = 0
    return (offset - np.pi + np.fmod(add_pos-offset+np.pi+x, 2*np.pi))
    """
    # original
    #return (offset -np.pi + (-offset+np.pi+x) % (2*np.pi))
    return (offset -np.pi + np.remainder(-offset + np.pi + x , 2*np.pi))


def fix2pi_skips(phase, sign='+', around=None, debug=0):
    """ ensure that phase monotonically increases (+) or decreases by
    adding units of 2Pi - Use modtwopi if you want to keep in that range.
    !!!! 2013 - fixed problem  - loss of two first points!
    >>> ph = fix2pi_skips([1,8,3,-2]); \
    print(', '.join(["{0:.3f}".format(p) for p in ph]))
    1.000, 1.717, 3.000, 4.283
    >>> ph = fix2pi_skips([4,11,7,1],around=None); \
    print(', '.join(["{0:.3f}".format(p) for p in ph]))
    4.000, 4.717, 7.000, 7.283
    >>> ph = fix2pi_skips([4,11,7,1],around=0); \
    print(', '.join(["{0:.3f}".format(p) for p in ph]))
    -2.283, -1.566, 0.717, 1.000

    New method is highly vectorised and much much faster, doesn't need 
    to know the sign.
    old method is still here (for now) but not very efficient!
    """
#    print(phase)
# had to do two steps if the jump is big - maybe should do more (or write something better)
    fixed_phase = np.insert(np.cumsum((np.diff(phase) < -np.pi)*2*np.pi)
                            +phase[1:],0,phase[0])
    fixed_phase = np.insert(np.cumsum((np.diff(phase) < -np.pi)*2*np.pi)
                            +phase[1:],0,phase[0])
    if debug>0: print(fixed_phase)
    fixed_phase = np.insert(np.cumsum((np.diff(fixed_phase) > np.pi)*-2*np.pi)
                            +fixed_phase[1:],0,fixed_phase[0])
    fixed_phase = np.insert(np.cumsum((np.diff(fixed_phase) > np.pi)*-2*np.pi)
                            +fixed_phase[1:],0,fixed_phase[0])
    if around != None:
        if np.std(fixed_phase)< 3: # use the average
            ref = np.average(fixed_phase)
        else:                      # use the first
            ref = fixed_phase[0]

        fixed_phase = fixed_phase - 2*np.pi*int(0.5+(ref-around)/(2*np.pi))
    return(fixed_phase)

# this code is never accessed now - maybe keep for tests later
# this was original - only good for a few skips - otherwise slow
    phase = copy(phase)   # to avoid overwriting original
    for i in 1+arange(len(phase[1:])):
        if sign=='+':
            if pyfusion.settings.VERBOSE>3: print("fix2pi phase %d before = %4.4f" % (i, phase[i]))
            if phase[i]<phase[i-1]: 
                for j in range(i,len(phase)): phase[j]+=2*np.pi
        else: 
            if phase[i]>phase[i-1]:
                for j in range(i,len(phase)): phase[j]-=2*np.pi
#    print(phase)
    return(phase)

from numpy import array, arange, min, max
def decimate(data, fraction=None, limit=None):
    """ reduce the number of items to a limit or by a fraction
    """
    if (fraction is None and limit is None):
        limit=2000
    if fraction != None:
        step = max([int(1/fraction),1])
    else:
        step = max([int(len(data)/limit),1])
    return(array(data)[arange(0,len(data), step)])        

if __name__ == "__main__":
    import doctest
    doctest.testmod()

