"""
Test lasso in an "add-on" form - can be used on existing graphs
without planning ahead.  This usually means things are a little
less objectified than a pure example.  In the simplest incarnation,
the points are passed simply through the axes object.
Modified the original to draw small triangles (unfilled) for speed.

In a more sophisticated version, the clusters could be passed 
(or even the fs) as well as the axis, so that fs info could be used.

Put a limit on the number of triangles drawn (max_collection) less printout
and nicer to use.
                               From lasso_demo.py
"""
import pyfusion
#from pyfusion.datamining.clustering.core import get_fs_in_set, FluctuationStructure, ClusterDataSet
from matplotlib.widgets import Lasso
import matplotlib.mlab
from matplotlib.nxutils import points_inside_poly
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection

from matplotlib.pyplot import figure, show
from numpy import nonzero, array, abs
from numpy.random import rand

class Datum:
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('green')
    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include: self.color = self.colorin
        else: self.color = self.colorout


class LassoManager:
    def __init__(self, ax, data, fs_list, plkwargs, max_collection=500):
        if pyfusion.VERBOSE>2:
            print("init, fs_list size=%d" % (len(fs_list)))
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data
        self.fs_list = fs_list
        self.plkwargs = plkwargs

        self.Nxy = len(data)

        facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]
        fig = ax.figure
        # here we define the polygons that indicate that a point is registered
        # - use empty triangles for simplicity and speed
        # skip altogether if there are too many
        # not sure how much this saves - each point still is a datum
        if (self.Nxy<max_collection):
            self.collection = RegularPolyCollection(
                3, 0, sizes=(100,),facecolors='', edgecolors=facecolors,
                offsets = self.xys, transOffset = ax.transData)

            ax.add_collection(self.collection)
        else: self.collection = None    

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        print("connected cid=%d" % (self.cid) )

        self.ind = None

    def callback(self, verts):
        ind = nonzero(points_inside_poly(self.xys, verts))[0]
        if pyfusion.VERBOSE>2: print len(self.xys)
        if pyfusion.VERBOSE>0: 
            print("Try to match the following points to %d fs in list:" % (len(self.fs_list)))
            print [self.xys[i] for i in ind]
            
        if self.collection != None:    
            edgecolors = self.collection.get_edgecolors()
            for i in range(self.Nxy):
                if i in ind:
                    edgecolors[i] = Datum.colorin
                else:
                    edgecolors[i] = Datum.colorout

        print(self)        
        xarr = [xy[0] for xy in self.xys]
        yarr = [xy[1] for xy in self.xys]
        fstarr = array(self.fs_list['t_mid'])
        fsfarr = array(self.fs_list['freq'])
        if pyfusion.VERBOSE>5:
            print("xarr = %s, fsfarr = %s, fstarr=%s, ind = %s" % (xarr, fsfarr, fstarr, ind))
        for i in ind:
            match = (abs(xarr[i] - fstarr) + abs(yarr[i] - fsfarr))< 1e-3
            matchinds = match.nonzero()[0]
            if pyfusion.VERBOSE>5: print("matches", matchinds)
            if len(matchinds)>6:
                print('only showing first 6...')
                matchinds = matchinds[0:10]
            for m in matchinds: 
                #self.fs_list[m].plot(**self.plkwargs)
                print("shot {0}, time {1} s".format(self.fs_list['shot'][m],self.fs_list['t_mid'][m]))
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso
        self.ind = ind
    def onpress(self, event):
        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        if pyfusion.VERBOSE>7: print event.xdata
        # left button is normal - right does lasso - but this seesm to hang - crash?
        ## if event.button == 1: self.canvas.button_press_event(event)  
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

import pylab as pl
import  numpy as np


lass_man = None

def lasso_fs_disconnect():
    """ stops the lines being drawn, as well as being neat)
    """
    lass_man.canvas.mpl_disconnect(lass_man.cid)

def lasso_fs_init( fs_list, ax=None, all=True, plkwargs={}, matchkwargs={}, max_collection=500):
    """ Set up to select fs points by lassoing them.  The widget coding is above, this is the call
    that makes and breaks the connection.  See examples/Boyds/test_find_closest_cluster.py for a
    realistic example, and examples/Boyds/test_lasso_fs.py for a simple one.
         
         ax is axes object to use, 
         plkwargs is a dict containing kw args for plot of fs
         matchkwargs controls the matching of points with fs - e.g.
             {'xaxis': 'time', yaxis: 'frequency', 'tolerance': 1e-3, 
             'call_function' : 'self.plot'}
         Example for higher precision in e and a2
             lasso_fs_init(fs_list=fs_lst,plkwargs={'fmt':"%.4f"})

    """
    global lass_man
    if ax is None: ax=pl.gca()
    chldrn = ax.get_children()
    xys = []
    # looks like we are checking out all the points on the current "axes"
    # and registering them as possible lasso candidates.
    # this could get messy with large numbers of points plotted.
    for ch in chldrn:
        if ch.__dict__.has_key('_xy'):
            if pyfusion.VERBOSE>1: 
                print ch
            xyarr = ch.__dict__['_xy'] # actually a masked array - look at mask?
            for xy in xyarr: xys.append([xy[0],xy[1]])
            if pyfusion.VERBOSE>1:
                print('xyarr data %d points' % (len(xyarr)))
            if pyfusion.VERBOSE>3:
                print('xyarr=%s' %(xyarr))
#    data = [Datum(*xydat) for xydat in xy.data]  this form if a masked array
    if pyfusion.VERBOSE>2:
        print("registering %d points" % (len(xys)))
    data = [Datum(*xydat) for xydat in xys]

    if lass_man != None: lass_man.canvas.mpl_disconnect(lass_man.cid)
    lman = LassoManager(ax, data, fs_list, plkwargs,max_collection=max_collection)
    lass_man = lman
    show()
    print('"lasso" your points - lasso_fs_disconnect() to restore normal graphics')
    return(lman)

if __name__ == '__main__':

# test program
    import pyfusion

    fs_list = pyfusion.session.query(FluctuationStructure).join(['svd','timesegment','shot']).all()
    print("Found %d fs " % (len(fs_list)))

    if len(fs_list)>10: fs_list = fs_list[0:9]
    pl.plot(fs_list['t_mid'],fs_list['freq'], marker='o',linestyle='')
    
#    lasso_fs_init(fs_list=fs_list)
