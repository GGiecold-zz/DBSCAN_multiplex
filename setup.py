# -*- coding: utf-8 -*-
#!/usr/bin/env python


# DBSCAN_multiplex/setup.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""Setup script for DBSCAN_multiplex, a fast and memory-efficient implementation of DBSCAN 
(Density-Based Spatial Clustering of Appplications with Noise). 
The gain is especially outstanding for applications involving multiple rounds of down-sampling
and clustering from a common dataset.

References
----------
* Ester, M., Kriegel, H.-P., Sander, J. and Xu, X., "A Density-Based Algorithm for Discovering Clusters in Large Spatial
Databases with Noise". 
In: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), pp. 226–231. 1996
* Kriegel, H.-P., Kroeger, P., Sander, J. and Zimek, A., "Density-based Clustering". 
In: WIREs Data Mining and Knowledge Discovery, 1, 3, pp. 231–240. 2011
"""


import codecs
from os import path
from sys import version
from distutils.core import setup


if version < '2.2.3':
    from distutils.dist import DistributionMetadata
    
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None
    
    
here = path.abspath(path.dirname(__file__))

try:
    import pypandoc
    
    z = pypandoc.convert('README.md', 'rst', format = 'markdown')
    
    with open(path.join(here, 'README'), 'w') as f:
        f.write(z)
        
    with codecs.open(path.join(here, 'README'), encoding = 'utf-8') as f:
        long_description = f.read()
except:
    print("WARNING: 'pypandoc' module not found: could not convert from Markdown to RST format")
    long_description = ''


setup(name = 'DBSCAN_multiplex',
      version = '1.5',
      
      description = 'Fast and memory-efficient DBSCAN clustering,'
                    'possibly on various subsamples out of a common dataset',
      long_description = long_description,
                    
      url = 'https://github.com/GGiecold/DBSCAN_multiplex',
      download_url = 'https://github.com/GGiecold/DBSCAN_multiplex',
      
      author = 'Gregory Giecold',
      author_email = 'g.giecold@gmail.com',
      maintainer = 'Gregory Giecold',
      maintainer_email = 'ggiecold@jimmy.harvard.edu',
      
      license = 'MIT License',
      
      py_modules = ['DBSCAN_multiplex'],
      platforms = ('Any',),
      requires = ['numpy (>=1.9.0)', 'sklearn', 'tables'],
                          
      classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Mathematics', ],
                   
      keywords = 'machine-learning clustering',
)
