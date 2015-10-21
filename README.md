# DBSCAN_multiplex
A fast and memory-efficient implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

It is especially suited for multiple rounds of down-sampling and clustering from a joint dataset: after an initial overhead O(N log(N)), each subsequent run of clustering will have O(N) time complexity. 

As illustrated by a doctest embedded in the present module's docstring, on a dataset of 15,000 samples and 47 features, DBSCAN_multiplex performs 50 rounds of sub-sampling and clustering in about 4 minutes, whereas Scikit-learn's implementation of DBSCAN performs the same task in more than 28 minutes. 
