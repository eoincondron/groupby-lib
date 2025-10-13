groupby-lib Documentation
============================

**groupby-lib** is a high-performance extension package for pandas that provides optimized groupby operations using NumPy arrays and Numba's just-in-time compilation.

The package is designed to work seamlessly with pandas DataFrames and Series while providing significant performance improvements for various array types.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_reference
   examples
   contributing

Features
--------

* **Fast GroupBy Operations**: Optimized implementations using Numba JIT compilation
* **Multiple Array Support**: Works with pandas Series, NumPy arrays, and other array-like objects  
* **Seamless Integration**: Drop-in replacement for pandas groupby operations
* **Performance Focused**: Significant speed improvements for large datasets
* **Extensible**: Easy to add custom aggregation functions

Quick Start
-----------

Install from PyPI::

    pip install groupby-lib

Or from conda-forge::

    conda install -c conda-forge groupby-lib

Basic usage::

    from groupby_lib.groupby import GroupBy
    import pandas as pd
    import numpy as np
    
    # Create sample data
    key = pd.Series([0, 0, 1, 1, 2, 2])
    values = pd.Series([1, 2, 3, 4, 5, 6])
    
    # Create GroupBy object and compute sum
    gb = GroupBy(key)
    result = gb.sum(values)
    print(result.values)  # [3, 7, 11]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

