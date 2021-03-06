# data should be prepared by example1.py

library('foreign')  	# routines to read ‘foreign’ data
ne_p <- read.arff('ne_profile.arff')
library('cluster') 	             # clustering library 
(kn <- kmeans(ne_p, 3))  # kmeans clustering assuming 3 clusters. Extra () prints result.

# second step - the full dataset, contains metadata shot and t_mid

ne_pall <- read.arff('ne_profile_all.arff')
(ka <- kmeans(ne_pall[c(grepl("ne_prof",colnames(ne_pall)))], 3))  # [c..] selects columns 
plot(ne_pall[c("shot", "t_mid")],col=ka$cluster)  	#  plot time vs shot, coloured by cluster.

# really need R procedure for plotting  1D lines according to cluster.