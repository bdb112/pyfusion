import numpy as np
def rotate(x, offs=3):
    """ >>> rotate([1,2,3,4,5,6],2)
         [3,4,5,6,1,2]
    """
    xlist = x.tolist()
    return(np.array(xlist[offs:] + xlist[0:offs]))

def boxcar(sig=None, period=167, maxnum=99999999, debug=0):
    """
    period    # actual period in samples
    """
    # make sure that any MDS data has been converted to a true nd.array
    # test for presence of .value_of()  because nd.arrays have a (different) .data() method!xs
    if hasattr(sig, 'value_of'):  # otherwise will get 1 extra element???
        sig = sig.data()
    ip = int(period)  # integer part of the period
    numcyc = len(sig) // ip - 2   # take off 2 for a safety margin
    numcyc = numcyc - numcyc//ip # and allow for 1 extra point lost each cycle
    if numcyc > maxnum:
        numcyc = min(maxnum, numcyc)  # can reduce number as we do the tuning.
        print('truncating to {numcyc} cycles'.format(numcyc=numcyc))

    ar2d = []
    for cyc in range(numcyc):
        offs = int(cyc * (period - ip))  # offset grows a little each time
        ar2d.append(sig[cyc * ip + offs: (cyc + 1) * ip + offs])

    accum = np.mean(ar2d,axis=0)
    if debug:
        plt.plot(accum)
    return(accum)
    
