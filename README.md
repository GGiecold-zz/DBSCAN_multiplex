# DBSCAN_multiplex
A fast and memory-efficient implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

It is especially suited for multiple rounds of down-sampling and clustering from a joint dataset: after an initial overhead O(N log(N)), each subsequent run of clustering will have O(N) time complexity.
