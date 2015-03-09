""" Test code for lasso_fs developed from test_fs_plot 
for FluctuationStructure.plot() 
Example: 
1/ run with no plot, it will show all fs in current database
2/ run after plot_N_clusters,  - all fs will be added to graph, and can lasso
3/ run after any other plot of f vs t

plot_points=0    suppress plotting of points
fs_lst           an list of flucstrucs to restrict attention to
show_svs
maxpts           the maximum number of fluctstrucs read
"""
import pyfusion
import pylab as pl

#from pyfusion.datamining.clustering.core import FluctuationStructure

maxpts=2000
show_svs= True
fs_lst= None
plot_points=True
hold=True

import pyfusion.utils
exec(pyfusion.utils.process_cmd_line_args())

# indiscriminate, so works on any database
# in a real application, need to be more discriminating
# probably should use the same subset as has been plotted.
# this is simply acheived by supplying the fs_lst
dds = {} # subset
for key in dd.keys():
    dds.update({key:dd[key][ind]})
if fs_lst is None:
#    fsqry = pyfusion.session.query(FluctuationStructure).limit(maxpts)
#    if fsqry.count() == 0: raise LookupError, 'No fs found in current database'
    
    fs_lst = dds

#pl.figure() # no! want to overlay 
#pl.ion()   # needed for some ipython/matplotlib configs

# what does this do? # linestyle='')
if plot_points: pl.plot(fs_lst['t_mid'],  fs_lst['freq'], ',k',hold=hold)

from pyfusion.visual.lasso_fs_init import lasso_fs_init, lasso_fs_disconnect
lasso_fs_init(fs_list = dds, max_collection=0)

if len(fs_lst) == maxpts: 
    print("**** Warning - only %d fs retrieved - increase maxpts*****" 
          % maxpts)
