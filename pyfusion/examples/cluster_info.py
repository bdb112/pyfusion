# show the extent of population inside one_rms
import pyfusion.clustering.clustering as clust
import numpy as np

cluster_data = clust.feature_object(filename='/data/datamining/shaun/10May2013_bb.pickle')
cd=cluster_data
co=cd.clustered_objects[-1]
ml=cluster_data.clustered_objects[-1].make_mode_list(min_kappa=0);[mm.name for mm in ml]
ml=co.make_mode_list(min_kappa=0);[mm.name for mm in ml]
print('fractions of one_rms<1')

for m in ml: 
    numless = np.where(ml[m.id].one_rms(cd.instance_array[co.members(m.id)])<1)[0]
    numlessall = np.where(ml[m.id].one_rms(cd.instance_array)<1)[0]
    print('{nm:6s}: {nl:10d}, {f:7.3g}% of its members, {fall:.2f}% of all'
          .format(nm=m.name, nl = len(numless),
                  f=100*(len(numless)/
                         float(len(co.members(m.id)))),
                  fall=100*len(numless)/
                  float(len(cd.instance_array))))
