""" Goes with realtime_mode.py
"""
import pickle
from pyfusion.examples import Mode
from numpy import sqrt
import pyfusion.clustering.clustering as clust
cluster_data = clust.feature_object(filename='/data/datamining/shaun/10May2013.pickle')

co=cluster_data.clustered_objects[-1]; co.plot_clusters_phase_lines()
m6=Mode(name='m6', N=4, NN=400,cc=co.cluster_details['means'][6],csd=1/sqrt(co.cluster_details['variance'][6]))
m7=Mode(name='m7', N=5, NN=500,cc=co.cluster_details['means'][7],csd=1/sqrt(co.cluster_details['variance'][7]))
m13=Mode(name='m13', N=4, NN=400,cc=co.cluster_details['means'][13],csd=1/sqrt(co.cluster_details['variance'][13]))
m14=Mode(name='m14', N=4, NN=400,cc=co.cluster_details['means'][14],csd=1/sqrt(co.cluster_details['variance'][14]))
m4=Mode(name='m4', N=5, NN=500,cc=co.cluster_details['means'][4],csd=1/sqrt(co.cluster_details['variance'][4]))
srh_thesis = [m4, m6, m7, m13, m14]
pickle.dump(srh_thesis,file('srh_thesis.pickle','w'))

