#!/usr/bin/env python


# DBSCAN_multiplex/DBSCAN_multiplex.py


# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""A fast and memory-efficient implementation of DBSCAN 
(Density-Based Spatial Clustering of Applications with Noise) 
for repeated rounds of random subsamplings and clusterings from a data-set. 

After a linear overhead, each call to DBSCAN has O(N) time complexity,
instead of the time complexity O(N log(N)) associated with most other implementations of DBSCAN.

In addition, unlike Scikit-learn's implementation, by relying on an HDF5 data structure this version of DBSCAN
can deal with large data-sets without saturating the machine's memory resources.

This module is especially suitable for repeated runs of clusterings on subsamples of a data-set. 
This incurs an initial overhead but then each subsequent call to DBSCAN has O(N) time complexity. 
In comparison, the standard implementation of DBSCAN runs in O(N log(N)).
       
For instance, consider a small data-set of 15,000 samples, generated as a normally distributed matrix of 47 features:

        
>>> data = np.random.randn(15000, 47)

        
Set "minPts" to 3. Repeat the following procedure 50 times: 
i)  take a random subsample of those points (say, 90 % from the whole data-set);
ii) perform DBSCAN clustering on this subsample.

        
>>> N_iterations = 50
>>> N_sub = data.shape[0] * 9 / 10

>>> subsamples_matrix = np.zeros((N_iterations, N_sub), dtype = int)
>>> for i in xrange(N_iterations): subsamples_matrix[i] = np.random.choice(data.shape[0], N_sub, replace = False)
    
>>> beg_multiplex = time.time()
>>> eps, labels_matrix = DBSCAN(data, 3, subsamples_matrix = subsamples_matrix, verbose = False)
>>> end_multiplex = time.time()
        
>>> beg_sklearn = time.time()
>>> db = sklearn.cluster.DBSCAN(eps, 3)
>>> for i in xrange(N_iterations): y = db.fit_predict(data[subsamples_matrix[i]])
>>> end_sklearn = time.time()
        
>>> assert (end_sklearn - beg_sklearn) > 6exit( * (end_multiplex - beg_multiplex)
True

       
On the laptop where this code was tested (Asus Zenbook with 8GiB of RAM and 
an Intel Core M processor), the implementation herewith took 
about 188 + 50 * 1.2 seconds to complete, whereas the version provided by Scikit-learn needed 
at least 50 * 34 seconds, i.e. slightly over 4 minutes versus more than 28 minutes, illustrating 
the overwelming advantage of the present implementation (not even taking into account the time needed for determining
a suitable value for epsilon, which accounts for about 39 seconds out of those 188 seconds).
        
The benefits of our implementation are even more significant for a larger number of samples. 
For instance, for a matrix of 50000 samples and 47 features, 50 rounds of clustering of random subsamples 
of size 80% would take about 1608 + 50 * 1.3 = 1673 seconds for this version, 
versus 50 * 653 = 32650 seconds for Scikit-learn.
"""


import gc
import platform
import re
import subprocess
import tables
from tempfile import NamedTemporaryFile
import time
import warnings

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph 
from sklearn.utils import check_random_state


np.seterr(invalid = 'ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)


__all__ = ['DBSCAN', 'load', 'shoot']


def memory():
    """Determine the machine's memory specifications.

    Returns
    -------
    mem_info : dictonary
        Holds the current values for the total, free and used memory of the system.

    """

    mem_info = {}

    if platform.linux_distribution()[0]:
        with open('/proc/meminfo') as file:
            c = 0
            for line in file:
                lst = line.split()
                if str(lst[0]) == 'MemTotal:':
                    mem_info['total'] = int(lst[1])
                elif str(lst[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    c += int(lst[1])
            mem_info['free'] = c
            mem_info['used'] = (mem_info['total']) - c
    elif platform.mac_ver()[0]:
        ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0]
        vm = subprocess.Popen(['vm_stat'], stdout=subprocess.PIPE).communicate()[0]

        # Iterate processes
        process_lines = ps.split('\n')
        sep = re.compile('[\s]+')
        rss_total = 0  # kB
        for row in range(1, len(process_lines)):
            row_text = process_lines[row].strip()
            row_elements = sep.split(row_text)
            try:
                rss = float(row_elements[0]) * 1024
            except:
                rss = 0  # ignore...
            rss_total += rss

        # Process vm_stat
        vm_lines = vm.split('\n')
        sep = re.compile(':[\s]+')
        vm_stats = {}
        for row in range(1, len(vm_lines) - 2):
            row_text = vm_lines[row].strip()
            row_elements = sep.split(row_text)
            vm_stats[(row_elements[0])] = int(row_elements[1].strip('\.')) * 4096

        mem_info['total'] = rss_total
        mem_info['used'] = vm_stats["Pages active"]
        mem_info['free'] = vm_stats["Pages free"]
    else:
        raise('Unsupported Operating System.\n')
        exit(1)

    return mem_info


def get_chunk_size(N, n):
    """Given a dimension of size 'N', determine the number of rows or columns 
       that can fit into memory.

    Parameters
    ----------
    N : int
        The size of one of the dimension of a two-dimensional array.  

    n : int
        The number of times an 'N' by 'chunks_size' array can fit in memory.

    Returns
    -------
    chunks_size : int
        The size of a dimension orthogonal to the dimension of size 'N'. 
    """

    mem_free = memory()['free']
    if mem_free > 60000000:
        chunks_size = int(((mem_free - 10000000) * 1000) / (4 * n * N))
        return chunks_size
    elif mem_free > 40000000:
        chunks_size = int(((mem_free - 7000000) * 1000) / (4 * n * N))
        return chunks_size
    elif mem_free > 14000000:
        chunks_size = int(((mem_free - 2000000) * 1000) / (4 * n * N))
        return chunks_size
    elif mem_free > 8000000:
        chunks_size = int(((mem_free - 1400000) * 1000) / (4 * n * N))
        return chunks_size
    elif mem_free > 2000000:
        chunks_size = int(((mem_free - 900000) * 1000) / (4 * n * N))
        return chunks_size
    elif mem_free > 1000000:
        chunks_size = int(((mem_free - 400000) * 1000) / (4 * n * N))
        return chunks_size
    else:
        raise MemoryError("\nERROR: DBSCAN_multiplex @ get_chunk_size:\n"
                          "this machine does not have enough free memory "
                          "to perform the remaining computations.\n")


def load(hdf5_file_name, data, minPts, eps = None, quantile = 50, subsamples_matrix = None, samples_weights = None, 
metric = 'minkowski', p = 2, verbose = True):
    """Determines the radius 'eps' for DBSCAN clustering of 'data' in an adaptive, data-dependent way.

    Parameters
    ----------
    hdf5_file_name : file object or string
        The handle or name of an HDF5 data structure where any array needed for DBSCAN
        and too large to fit into memory is to be stored.

    data : array of shape (n_samples, n_features)
        An array of features retained from the data-set to be analysed. 
        Subsamples of this curated data-set can also be analysed by a call to DBSCAN by providing an appropriate 
        list of selected samples labels, stored in 'subsamples_matrix' (see below).

    subsamples_matrix : array of shape (n_runs, n_subsamples), optional (default = None)
        Each row of this matrix contains a set of indices identifying the samples selected from the whole data-set
        for each of 'n_runs' independent rounds of DBSCAN clusterings.

    minPts : int
        The number of points within an epsilon-radius hypershpere for the said region to qualify as dense.

    eps : float, optional (default = None)
        Sets the maximum distance separating two data-points for those data-points to be considered 
        as part of the same neighborhood.

    quantile : int, optional (default = 50)
        If 'eps' is not provided by the user, it will be determined as the 'quantile' of the distribution 
        of the k-nearest distances to each sample, with k set to 'minPts'.

    samples_weights : array of shape (n_runs, n_samples), optional (default = None)
        Holds the weights of each sample. A sample with weight greater than 'minPts' is guaranteed to be
        a core sample; a sample with negative weight tends to prevent its 'eps'-neighbors from being core. 
        Weights are absolute and default to 1.

    metric : string or callable, optional (default = 'euclidean')
        The metric to use for computing the pairwise distances between samples 
        (each sample corresponds to a row in 'data'). If metric is a string or callable, it must be compatible 
        with metrics.pairwise.pairwise_distances.

    p : float, optional (default = 2)
        If a Minkowski metric is used, 'p' determines its power.

    verbose : Boolean, optional (default = True)
        Whether to display messages reporting the status of the computations and the time it took 
        to complete each major stage of the algorithm. 

    Returns
    -------
    eps : float
        The parameter of DBSCAN clustering specifying if points are density-reachable. 
        This is either a copy of the value provided at input or, if the user did not specify a value of 'eps' at input, 
        the return value if the one determined from k-distance graphs from the data-set.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
    """
    
    data = np.array(data, copy = False)
    if data.ndim > 2:
        raise ValueError("\nERROR: DBSCAN_multiplex @ load:\n" 
                         "the data array is of dimension %d. Please provide a two-dimensional "
                         "array instead.\n" % data.ndim)

    if subsamples_matrix is None:
        subsamples_matrix = np.arange(data.shape[0], dtype = int)
        subsamples_matrix = subsamples_matrix.reshape(1, -1)
 
    else:
        subsamples_matrix = np.array(subsamples_matrix, copy = False)

    if subsamples_matrix.ndim > 2:
        raise ValueError("\nERROR: DBSCAN_multiplex @ load:\n"
                         "the array of subsampled indices is of dimension %d. "
                         "Please provide a two-dimensional array instead.\n" % subsamples_matrix.ndim)

    if (data.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(data.sum()) and not np.all(np.isfinite(data))):
        raise ValueError('\nERROR: DBSCAN_multiplex @ load:\n'
                         'the data vector contains at least one infinite or NaN entry.\n')

    if (subsamples_matrix.dtype.type is np.int_ and not np.isfinite(subsamples_matrix.sum()) and not np.all(np.isfinite(subsamples_matrix))):
        raise ValueError('\nERROR: DBSCAN_multiplex @ load:\n' 
                         'the array of subsampled indices contains at least one infinite or NaN entry.\n')

    if not np.all(subsamples_matrix >= 0):
        raise ValueError('\nERROR: DBSCAN_multiplex @ load:\n'
                         'the sampled indices should all be positive integers.\n') 

    N_samples = data.shape[0]
    N_runs, N_subsamples = subsamples_matrix.shape

    if N_subsamples > N_samples:
        raise ValueError('\nERROR: DBSCAN_multiplex @ load:\n'
                         'the number of sampled indices cannot exceed the total number of samples in the whole data-set.\n')

    for i in xrange(N_runs):
        subsamples_matrix[i] = np.unique(subsamples_matrix[i])
 
    if not isinstance(minPts, int):
        raise TypeError("\nERROR: DBSCAN_multiplex @ load:\n"
                        "the parameter 'minPts' must be an integer.\n")

    if minPts < 2:
        raise ValueError("\nERROR: DBSCAN_multiplex @ load:\n"
                         "the value of 'minPts' must be larger than 1.\n")        

    if eps is None:
        # Determine the parameter 'eps' as the median of the distribution
        # of the maximum of the minPts-nearest neighbors distances for each sample.
        if verbose:
            print("INFO: DBSCAN_multiplex @ load:\n"
                  "starting the determination of an appropriate value of 'eps' for this data-set"
                  " and for the other parameter of the DBSCAN algorithm set to {minPts}.\n"
                  "This might take a while.".format(**locals()))

        beg_eps = time.time()

        quantile = np.rint(quantile)
        quantile = np.clip(quantile, 0, 100)

        k_distances = kneighbors_graph(data, minPts, mode = 'distance', metric = metric, p = p).data
 
        radii = np.zeros(N_samples, dtype = float)
        for i in xrange(0, minPts):
            radii = np.maximum(radii, k_distances[i::minPts]) 
             
        if quantile == 50:     
            eps = round(np.median(radii, overwrite_input = True), 4)
        else:
            eps = round(np.percentile(radii, quantile), 4)

        end_eps = time.time()

        if verbose:
            print("\nINFO: DBSCAN_multiplex @ load:\n"
                  "done with evaluating parameter 'eps' from the data-set provided."
                  " This took {} seconds. Value of epsilon: {}.".format(round(end_eps - beg_eps, 4), eps))

    else:
        if not (isinstance(eps, float) or isinstance(eps, int)):
            raise ValueError("\nERROR: DBSCAN_multiplex @ load:\n"
                             "please provide a numeric value for the radius 'eps'.\n")

        if not eps > 0.0:
            raise ValueError("\nERROR: DBSCAN_multiplex @ load:\n"
                             "the radius 'eps' must be positive.\n")

        eps = round(eps, 4)

    # For all samples with a large enough neighborhood, 'neighborhoods_indices' 
    # and 'neighborhoods_indptr' help us find the neighbors to every sample. Note
    # that this definition of neighbors leaves the original point in,
    # which will be considered later.
    if verbose:
       print("\nINFO: DBSCAN_multiplex @ load:\n"
             "identifying the neighbors within an hypersphere of radius {eps} around each sample,"
             " while at the same time evaluating the number of epsilon-neighbors for each sample.\n"
             "This might take a fair amount of time.".format(**locals()))

    beg_neigh = time.time()

    fileh = tables.open_file(hdf5_file_name, mode = 'r+')
    DBSCAN_group = fileh.create_group(fileh.root, 'DBSCAN_group')

    neighborhoods_indices = fileh.create_earray(DBSCAN_group, 'neighborhoods_indices', tables.Int32Atom(), (0,), 
                                                'Indices array for sparse matrix of neighborhoods', 
                                                expectedrows = int((N_samples ** 2) / 50))

    # 'neighborhoods_indptr' is such that for each of row i of the data-matrix 
    # neighborhoods_indices[neighborhoods_indptr[i]:neighborhoods_indptr[i+1]]
    # contains the column indices of row i from the array of 
    # 'eps'-neighborhoods.
    neighborhoods_indptr = np.zeros(1, dtype = np.int64)

    # For each sample, 'neighbors_counts' will keep a tally of the number 
    # of its  neighbors within a hypersphere of radius 'eps'. 
    # Note that the sample itself is counted as part of this neighborhood.
    neighbors_counts = fileh.create_carray(DBSCAN_group, 'neighbors_counts', tables.Int32Atom(), (N_runs, N_samples), 
                                           'Array of the number of neighbors around each sample of a set of subsampled points', 
                                           filters = None)   

    chunks_size = get_chunk_size(N_samples, 3)
    for i in xrange(0, N_samples, chunks_size):
        chunk = data[i:min(i + chunks_size, N_samples)]

        D = pairwise_distances(chunk, data, metric = metric, p = p, n_jobs = 1)
            
        D = (D <= eps)

        if samples_weights is None:
            for run in xrange(N_runs):
                x = subsamples_matrix[run]
                M = np.take(D, x, axis = 1)

                legit_rows = np.intersect1d(i + np.arange(min(chunks_size, N_samples - i)), x, assume_unique = True)
                M = np.take(M, legit_rows - i, axis = 0)
                
                neighbors_counts[run, legit_rows] = M.sum(axis = 1)

                del M
        else:
            for run in xrange(N_runs):
                x = subsamples_matrix[run]

                M = np.take(D, x, axis = 1)

                legit_rows = np.intersect1d(i + np.arange(min(chunks_size, N_samples - i)), x, assume_unique = True)
                M = np.take(M, legit_rows - i, axis = 0)

                neighbors_counts[run, legit_rows] = np.array([np.sum(samples_weights[x[row]]) for row in M])

                del M

        candidates = np.where(D == True)

        del D

        neighborhoods_indices.append(candidates[1])

        _, nbr = np.unique(candidates[0], return_counts = True)
        counts = np.cumsum(nbr) + neighborhoods_indptr[-1]

        del candidates

        neighborhoods_indptr = np.append(neighborhoods_indptr, counts)

    fileh.create_carray(DBSCAN_group, 'neighborhoods_indptr', tables.Int64Atom(), (N_samples + 1,), 
                        'Array of cumulative number of column indices for each row', filters = None)
    fileh.root.DBSCAN_group.neighborhoods_indptr[:] = neighborhoods_indptr[:]

    fileh.create_carray(DBSCAN_group, 'subsamples_matrix', tables.Int32Atom(), (N_runs, N_subsamples), 
                        'Array of subsamples indices', filters = None)
    fileh.root.DBSCAN_group.subsamples_matrix[:] = subsamples_matrix[:]

    fileh.close()

    end_neigh = time.time()

    if verbose:
        print("\nINFO: DBSCAN_multiplex @ load:\n"
              "done with the neighborhoods. This step took {} seconds.".format(round(end_neigh - beg_neigh, 4)))

    gc.collect()

    return eps
    

def shoot(hdf5_file_name, minPts, sample_ID = 0, random_state = None, verbose = True): 
    """Perform DBSCAN clustering with parameters 'minPts' and 'eps'
        (as determined by a prior call to 'load' from this module). 
        If multiple subsamples of the dataset were provided in a preliminary call to 'load', 
        'sample_ID' specifies which one of those subsamples is to undergo DBSCAN clustering. 
         
    Parameters
    ----------
    hdf5_file_name : file object or string
        The handle or name of an HDF5 file where any array needed for DBSCAN and too large to fit into memory 
        is to be stored. Procedure 'shoot' relies on arrays stored in this data structure by a previous
        call to 'load' (see corresponding documentation)

    sample_ID : int, optional (default = 0)
        Identifies the particular set of selected data-points on which to perform DBSCAN.
        If not subsamples were provided in the call to 'load', the whole dataset will be subjected to DBSCAN clustering.

    minPts : int
        The number of points within a 'eps'-radius hypershpere for this region to qualify as dense.

    random_state: np.RandomState, optional (default = None)
        The generator used to reorder the samples. If None at input, will be set to np.random.

    verbose : Boolean, optional (default = True)
        Whether to display messages concerning the status of the computations and the time it took to complete 
        each major stage of the algorithm.

    Returns
    -------
    core_samples : array of shape (n_core_samples, )
        Indices of the core samples.

    labels : array of shape (N_samples, ) 
        Holds the cluster labels of each sample. The points considered as noise have entries -1. 
        The points not initially selected for clustering (i.e. not listed in 'subsampled_indices', if the latter
        has been provided in the call to 'load' from this module) are labelled -2.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
    """       
        
    fileh = tables.open_file(hdf5_file_name, mode = 'r+')

    neighborhoods_indices = fileh.root.DBSCAN_group.neighborhoods_indices
    neighborhoods_indptr = fileh.root.DBSCAN_group.neighborhoods_indptr[:]

    neighbors_counts = fileh.root.DBSCAN_group.neighbors_counts[sample_ID]
    subsampled_indices = fileh.root.DBSCAN_group.subsamples_matrix[sample_ID]

    N_samples = neighborhoods_indptr.size - 1
    N_runs, N_subsamples = fileh.root.DBSCAN_group.subsamples_matrix.shape

    if not isinstance(sample_ID, int):
        raise ValueError("\nERROR: DBSCAN_multiplex @ shoot:\n"
                         "'sample_ID' must be an integer identifying the set of subsampled indices "
                         "on which to perform DBSCAN clustering\n")    

    if (sample_ID < 0) or (sample_ID >= N_runs):
        raise ValueError("\nERROR: DBSCAN_multiplex @ shoot:\n"
                 "'sample_ID' must belong to the interval [0; {}].\n".format(N_runs - 1))
      
    # points that have not been sampled are labelled with -2
    labels = np.full(N_samples, -2, dtype = int)
    # among the points selected for clustering, 
    # all are initally characterized as noise
    labels[subsampled_indices] = - 1
    
    random_state = check_random_state(random_state)

    core_samples = np.flatnonzero(neighbors_counts >= minPts)
 
    index_order = np.take(core_samples, random_state.permutation(core_samples.size))

    cluster_ID = 0

    # Look at all the selected samples, see if they qualify as core samples
    # Create a new cluster from those core samples
    for index in index_order:
        if labels[index] not in {-1, -2}:
            continue

        labels[index] = cluster_ID

        candidates = [index]
        while len(candidates) > 0:
            candidate_neighbors = np.zeros(0, dtype = np.int32)
            for k in candidates:
                candidate_neighbors = np.append(candidate_neighbors, 
                                                neighborhoods_indices[neighborhoods_indptr[k]: neighborhoods_indptr[k+1]])
                candidate_neighbors = np.unique(candidate_neighbors)

            candidate_neighbors = np.intersect1d(candidate_neighbors, subsampled_indices, assume_unique = True)
                
            not_noise_anymore = np.compress(np.take(labels, candidate_neighbors) == -1, candidate_neighbors)
            
            labels[not_noise_anymore] = cluster_ID

            # Eliminate as potential candidates the points that have already 
            # been used to expand the current cluster by a trail 
            # of density-reachable points
            candidates = np.intersect1d(not_noise_anymore, core_samples, assume_unique = True) 
     
        cluster_ID += 1
    # Done with building this cluster. 
    # "cluster_ID" is now labelling the next cluster.

    fileh.close()

    gc.collect()

    return core_samples, labels


def DBSCAN(data, minPts, eps = None, quantile = 50, subsamples_matrix = None, samples_weights = None, 
metric = 'minkowski', p = 2, verbose = True):
    """Performs Density-Based Spatial Clustering of Applications with Noise,
        possibly on various subsamples or combinations of data-points extracted from the whole dataset, 'data'.
        
        If the radius 'eps' is not provided by the user, it will be determined in an adaptive, data-dependent way 
        by a call to 'load' from this module (see the corresponding documentation for more explanations).
        
        Unlike Scikit-learn's and many other versions of DBSCAN, this implementation does not experience failure 
        due to 'MemoryError' exceptions for large data-sets.
        Indeed, any array too large to fit into memory is stored on disk in an HDF5 data structure.
    
    Parameters
    ----------
    data : array of shape (n_samples, n_features)
        The data-set to be analysed. Subsamples of this curated data-set can also be analysed 
        by a call to DBSCAN by providing lits of selected data-points, stored in 'subsamples_matrix' (see below).

    subsamples_matrix : array of shape (n_runs, n_subsamples), optional (default = None)
        Each row of this matrix contains a set of indices identifying the samples selected from the whole data-set 
        for each of 'n_runs' independent rounds of DBSCAN clusterings.

    minPts : int
        The number of points within an epsilon-radius hypershpere for the said region to qualify as dense.

    eps : float, optional (default = None)
        Sets the maximum distance separating two data-points for those data-points to be considered 
        as part of the same neighborhood.

    quantile : int, optional (default = 50)
        If 'eps' is not provided by the user, it will be determined as the 'quantile' of the distribution
        of the k-nearest distances to each sample, with k set to 'minPts'.

    samples_weights : array of shape (n_runs, n_samples), optional (default = None)
        Holds the weights of each sample. A sample with weight greater than 'minPts' is guaranteed 
        to be a core sample; a sample with negative weight tends to prevent its 'eps'-neighbors from being core. 
        Weights are absolute and default to 1.

    metric : string or callable, optional (default = 'euclidean')
        The metric to use for computing the pairwise distances between samples
        (each sample corresponds to a row in 'data'). 
        If metric is a string or callable, it must be compatible with metrics.pairwise.pairwise_distances.

    p : float, optional (default = 2)
        If a Minkowski metric is used, 'p' denotes its power.

    verbose : Boolean, optional (default = True)
        Whether to display messages reporting the status of the computations and the time it took to complete
        each major stage of the algorithm. 
    
    Returns
    -------
    eps : float
        The parameter of DBSCAN clustering specifying if points are density-reachable. 
        This is relevant if the user chose to let our procedures search for a value of this radius as a quantile
        of the distribution of 'minPts'-nearest distances for each data-point.
    
    labels_matrix : array of shape (N_samples, ) 
        For each sample, specifies the identity of the cluster to which it has been
        assigned by DBSCAN. The points classified as noise have entries -1. The points that have not been
        considered for clustering are labelled -2.
        
    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
    """
        
    assert isinstance(minPts, int) or type(minPts) is np.int_
    assert minPts > 1

    if subsamples_matrix is None:
        subsamples_matrix = np.arange(data.shape[0], dtype = int)
        subsamples_matrix = subsamples_matrix.reshape(1, -1)
    else:
        subsamples_matrix = np.array(subsamples_matrix, copy = False)

    N_runs = subsamples_matrix.shape[0]
    N_samples = data.shape[0]

    labels_matrix = np.zeros((N_runs, N_samples), dtype = int)

    with NamedTemporaryFile('w', suffix = '.h5', delete = True, dir = './') as f:
        eps = load(f.name, data, minPts, eps, quantile, subsamples_matrix, samples_weights, metric, p, verbose)

        for run in xrange(N_runs):
            _, labels = shoot(f.name, minPts, sample_ID = run, verbose = verbose)
            labels_matrix[run] = labels

    return eps, labels_matrix


if __name__ == '__main__':
    
    import doctest
    import sklearn.cluster
    
    doctest.testmod()
