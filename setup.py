#!/usr/bin/env python




# DBSCAN_multiplex/setup.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


r"""Setup script for DBSCAN_multiplex, a fast and memory-efficient implementation of DBSCAN 
(Density-Based Spatial Clustering of Appplications with Noise). 
The gain is especially outstanding for applications involving multiple rounds of down-sampling
and clustering from a common dataset.
"""




#*********************************************************************************
#*********************************************************************************


from sys import version
from distutils.core import setup


#*********************************************************************************
#*********************************************************************************


if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None
    

setup(name = 'DBSCAN_multiplex',
      version = '1.0',
      description = 'Fast and memory-efficient DBSCAN clustering,'
                    'possibly on various subsamples out of a common dataset',
      url = 'https://github.com/GGiecold/DBSCAN_multiplex',
      download_url = 'https://github.com/GGiecold/DBSCAN_multiplex',
      
      author = 'Gregory Giecold',
      author_email = 'g.giecold@gmail.com',
      maintainer = 'Gregory Giecold',
      maintainer_email = 'ggiecold@jimmy.harvard.edu',
      
      py_modules = ['DBSCAN_multiplex'],
      platforms = ('Any',),
      requires = ['numpy (>=1.9.0)', 'sklearn', 'tables'],
                          
      classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',             
                   'License :: OSI Approved :: Python Software Foundation License',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX',
                   'Programming Language :: Python', ],
)


#*********************************************************************************
#*********************************************************************************


