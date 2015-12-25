DBSCAN_multiplex
----------------
----------------

Overview
--------

A fast and memory-efficient implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

It is especially suited for multiple rounds of down-sampling and clustering from a joint dataset: after an initial overhead O(N log(N)), each subsequent run of clustering will have O(N) time complexity. 

As illustrated by a doctest embedded in the present module's docstring, on a dataset of 15,000 samples and 47 features, on a Asus Zenbook laptop with 8 GiB of RAM and an Intel Core M processor, DBSCAN_multiplex processes 50 rounds of sub-sampling and clustering in about 4 minutes, whereas Scikit-learn's implementation of DBSCAN performs the same task in more than 28 minutes. 

Such a test can be performed quite conveniently on your machine: 
simply entering ```python DBSCAN_multiplex```in your terminal will prompt a doctest to start comparing the performance of the two afore-mentioned implementations of DBSCAN. 

This 7-fold gain in performance proved critical to the statistical learning application that prompted the design of this algorithm.

Installation and Requirements
-----------------------------

DBSCAN_multiplex requires Python 2.7 on a machine running any member of the Unix-like family of operating systems, along with the following packages and a few modules from the Standard Python Library:
* NumPy >= 1.9
* PyTables
* scikit-learn

You can install DBSCAN_multiplex from the official Python Package Index (PyPI) as follows:
* open a terminal window;
* type the command: ```pip install DBSCAN_multiplex```
The command listed above should automatically install or upgrade any missing or outdated dependency among those listed at the beginning of this section.

Usage and Example
-----------------

See the docstrings associated to each function of the DBSCAN_multiplex module for more information; in a Python interpreter console, they can be viewed by calling the built-in help system, e.g., ```help(DBSCAN_multiplex.load)```. 

The following few lines show how DBSCAN_multiplex can be used for clustering 50 randomly selected subsamples out of a common Gaussian distributed dataset. This situation arises in consensus clustering where one might want to obtain and then combine multiple vectors of cluster labels.

```
>>> import numpy as np
>>> import DBSCAN_multiplex as DB

>>> data = np.random.randn(15000, 7)
>>> N_iterations = 50
>>> N_sub = 9 * data.shape[0] / 10
>>> subsamples_matrix = np.zeros((N_iterations, N_sub), dtype = int)
>>> for i in xrange(N_iterations): 
        subsamples_matrix[i] = np.random.choice(data.shape[0], N_sub, replace = False)
>>> eps, labels_matrix = DB.DBSCAN(data, minPts = 3, subsamples_matrix = subsamples_matrix, verbose = True)
```

References
----------

* Ester, M., Kriegel, H.-P., Sander, J. and Xu, X., "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise". 
In: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), pp. 226–231. 1996
* Kriegel, H.-P., Kroeger, P., Sander, J. and Zimek, A., "Density-based Clustering". 
In: WIREs Data Mining and Knowledge Discovery, 1, 3, pp. 231–240. 2011
