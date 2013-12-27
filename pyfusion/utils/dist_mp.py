# from test_dist.py
# meant for multiprocessing distance calculations for clustering
#
# return for multithreading based on last example in
# http://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python

from __future__ import division
import threading
import numpy as np

class ThreadWorker():
    '''
    The basic idea is given a function create an object.
    The object can then run the function in a thread.
    It provides a wrapper to start it,check its status,and get data out the function.
    '''
    def __init__(self,func):
        self.thread = None
        self.data = None
        self.func = self.save_data(func)

    def save_data(self,func):
        '''modify function to save its returned data'''
        def new_func(*args, **kwargs):
            self.data=func(*args, **kwargs)

        return new_func

    def start(self,params, **kwargs):
        self.data = None
        if self.thread is not None:
            if self.thread.isAlive():
                return 'running' #could raise exception here

        #unless thread exists and is alive start or restart it
        self.thread = threading.Thread(target=self.func,args=params, kwargs=kwargs)
        if kwargs.has_key('debug') and (kwargs['debug']>0): print(kwargs)
        self.thread.start()
        return 'started'

    def status(self):
        if self.thread is None:
            return 'not_started'
        else:
            if self.thread.isAlive():
                return 'running'
            else:
                return 'finished'

    def get_results(self):
        if self.thread is None:
            return 'not_started' #could return exception
        else:
            if self.thread.isAlive():
                return 'running'
            else:
                return self.data

from time import sleep, time as seconds


try:
    from dist_nogil import dist
except ImportError, reason:
    if raw_input('auto compile cython code? [Y/n]').lower() != 'n':
        import pyximport; pyximport.install()
        from dist_nogil import dist

def dist_mp(cl_instance, instances, squared = None, averaged = None, threads=None, debug=0):
    """ multithreaded distance calculation using dist.pyx and nogil version
    set debug>=1 to to a comparison against single processor, GIL version
    >>> import numpy as np
    >>> x=dist_mp(np.random.random((14,)).astype(np.float32),np.random.random((1000,14)).astype(np.float32),debug=1,threads=3)

    """
    if threads is None:
        threads = 3

    n_workers = threads
    workers = []

    for w in range(n_workers): workers.append(ThreadWorker(dist))
    # allow 1 more, it may be short or empty, no prob.
    chunk = 1+len(instances)//n_workers

    st = seconds()
    for (w, worker) in enumerate(workers):
        #print worker.start([big[w*chunk:(w+1)*chunk],2])
        worker.start([cl_instance, instances[w*chunk:(w+1)*chunk,:]],
                     squared=squared, averaged=averaged, debug=debug)
#      NO! not this  kwargs=dict(squared=squared, debug=debug))


    while 'running' in  [worker.status() for worker in workers]:
        pass #print('waiting'),

    x = [] 
    for worker in workers: 
        x = np.append(x, worker.get_results())
        tm = seconds()-st
    if debug>0:
        from pyfusion.utils.dist import dist as dist_gil
        from numpy.testing import assert_array_almost_equal

        # extra \n to avoid thread output overwriting (mostly)
        if debug>1: print('\nchecking with safer, single thread version of dist_nogil')
        x1 = dist_gil(cl_instance, instances, 
                      averaged=averaged, squared=squared, debug=debug)
        assert_array_almost_equal(x, x1)
    print('took {dt:0.3g} sec for {w} workers'.format(dt=tm, w=n_workers))

    return(x)

"""
# this was the main program originally - should extract test 
code into a test routine.

n_workers = 3
workers = []

from dist_nogil import dist
for w in range(n_workers): workers.append(ThreadWorker(dist))

#for w in range(n_workers): workers.append(ThreadWorker(add))


import numpy as np

bignum = int(1e7)
careful = True
point = np.arange(14,dtype=np.float32)
if careful:
    points = np.random.random((bignum,14)).astype(np.float32)
else:
    points = np.ones((bignum,14),dtype=np.float32)


big = np.linspace(1,10,bignum)
#chunk = len(big)//n_workers
chunk = 1+len(points)//n_workers

st = seconds()
for (w, worker) in enumerate(workers):
    #print worker.start([big[w*chunk:(w+1)*chunk],2])
    worker.start([point, points[w*chunk:(w+1)*chunk,:]])


while 'running' in  [worker.status() for worker in workers]:
    pass #print('waiting'),

x = [] 
for worker in workers: 
    x = np.append(x, worker.get_results())
    tm = seconds()-st
print('took {dt:0.3g} sec for {w} workers'.format(dt=tm, w=n_workers))

print(x)

if careful:
    st1 = seconds()
    x1 = dist(point, points)
    t1 = seconds()-st1
    print('all close -> {ac}, {f:.2g}x faster'.
          format(ac=np.allclose(x, x1), f=t1/tm))
"""
if __name__ == "__main__":
    import doctest
    doctest.testmod()
"""
time dist(arange(10,dtype=float32),zeros((int(1e7),10),dtype=float32),threads=2, debug=2,squared=0,averaged=1)
"""
