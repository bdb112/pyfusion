"""
plots for clustering
"""

import pyfusion
import pylab as pl
from pyfusion.datamining.clustering.core import FluctuationStructure
from numpy import array,transpose, argsort

def plot_flucstrucs_for_shot(shot, diag_name, size_factor = 30.0, colour_factor = 30.0, frequency_range = [False,False], time_range=[False,False], savefile = ''):
    #TO DO: need to be able to seperate flucstrucs from different runs, etc...
    fs_list = pyfusion.session.query(FluctuationStructure).join(['svd','timesegment','shot']).join(['svd','diagnostic']).filter(pyfusion.Shot.shot == shot).filter(pyfusion.Diagnostic.name == diag_name).all()
    data = transpose(array([[f.svd.timebase[0], f.frequency, f.energy] for f in fs_list]))
    pl.scatter(data[0],data[1],size_factor*data[2], colour_factor*data[2])
    pl.grid(True)
    if not frequency_range[0]:
        frequency_range[0] = 0
    if not frequency_range[1]:
        nyquist = 0.5/(fs_list[0].svd.timebase[1]-fs_list[0].svd.timebase[0])
        frequency_range[1] = nyquist
    pl.ylim(frequency_range[0],frequency_range[1])
    if not time_range[0]:
        time_range[0] = min(data[0])
    if not time_range[1]:
        time_range[1] = max(data[0])
    pl.xlim(time_range[0], time_range[1])
    pl.title('Fluctuation structures for shot %d' %(shot))
    pl.xlabel('Time')
    pl.ylabel('Frequency')
    if savefile != '':
        try:
            pl.savefig(savefile)
        except:
            print 'could not save to filename %s. Please make sure filename ends with .png, .jpg, etc.'
            pl.show()
    else:
        pl.show()

def dendrogram(clusterdatasetname):
    pass

def simple_cluster_plot(clusterdatasetname, xlims = [0,1], ylims =[0,100000],  figurename = 'simple_cluster_plot.png'):
    from pyfusion.datamining.clustering.core import ClusterDataSet
    cluster_dataset = pyfusion.session.query(ClusterDataSet).filter_by(name=clusterdatasetname).one()
    cluster_sets =  cluster_dataset.clustersets
    cluster_sets_n_clusters = [i.n_clusters for i in cluster_sets]
    cluster_sets_n_clusters_argsort = argsort(cluster_sets_n_clusters)
    n_plots_horizontal = cluster_sets_n_clusters[cluster_sets_n_clusters_argsort[-1]]
    n_plots_vertical = len(cluster_sets_n_clusters_argsort)
    for cs_i, cs_el in enumerate(cluster_sets_n_clusters_argsort):
        for cli, cl in enumerate(cluster_sets[cs_el].clusters):
            plot_line_number = cs_i*n_plots_horizontal + cli + 1
            pl.subplot(n_plots_vertical, n_plots_horizontal, plot_line_number)
            data = transpose(array([[i.svd.timebase[0], i.frequency, i.energy] for i in cl.flucstrucs]))
            pl.plot(data[0],data[1],'.')
            pl.xlim(xlims[0], xlims[1])
            pl.ylim(ylims[0], ylims[1])
            if cs_i != len(cluster_sets_n_clusters_argsort)-1:
                pl.setp(pl.gca(), xticklabels=[])
            if cli != 0:
                pl.setp(pl.gca(), yticklabels=[])
            else:
                pl.ylabel('$N_{Cl} = %d$' %(cluster_sets_n_clusters[cluster_sets_n_clusters_argsort[cs_i]]))
    pl.savefig(figurename)
            