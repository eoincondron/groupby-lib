Getting Started
===============

Installation
------------

From PyPI
~~~~~~~~~~

Install the latest stable version from PyPI::

    pip install groupby-lib

From conda-forge
~~~~~~~~~~~~~~~~~

Install from conda-forge::

    conda install -c conda-forge groupby-lib

From Source
~~~~~~~~~~~

To install the development version from source::

    git clone https://github.com/eoincondron/groupby-lib.git
    cd groupby-lib
    pip install -e .[dev]

Basic Usage
-----------

Import the main classes::

    from groupby_lib.groupby import GroupBy
    import pandas as pd
    import numpy as np

Create sample data::

    # Sample data
    keys = pd.Series([0, 0, 1, 1, 2, 2, 0, 1])
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])

Basic GroupBy operations::

    # Create GroupBy object
    gb = GroupBy(keys)
    
    # Compute aggregations
    sums = gb.sum(values)
    means = gb.mean(values)
    counts = gb.count()
    
    print("Sums:", sums.values)     # [10,  15,  11]
    print("Means:", means.values)   # [3.33, 5.0,  5.5] 
    print("Counts:", counts.values) # [3,   2,   2]

Performance Benefits
--------------------

groupby-lib provides significant performance improvements over standard pandas groupby operations, especially for:

- Large datasets (>100K rows)
- Numeric operations (sum, mean, std, etc.)
- Repeated operations on the same grouping keys

Example performance comparison::

    import time
    import pandas as pd
    from groupby_lib.groupby import GroupBy
    
    # Large dataset
    n = 1_000_000
    keys = pd.Series(np.random.randint(0, 1000, n))
    values = pd.Series(np.random.randn(n))
    
    # Pandas groupby
    start = time.time()
    pandas_result = values.groupby(keys).sum()
    pandas_time = time.time() - start
    
    # groupby-lib groupby
    start = time.time()
    gb = GroupBy(keys)
    kungfu_result = gb.sum(values)
    kungfu_time = time.time() - start
    
    print(f"Pandas: {pandas_time:.3f}s")
    print(f"groupby-lib: {kungfu_time:.3f}s")
    print(f"Speedup: {pandas_time/kungfu_time:.1f}x")

Requirements
------------

- Python >= 3.10
- NumPy >= 1.19.0
- pandas >= 1.3.0
- numba >= 0.56.0
- polars >= 0.15.0 (optional, for enhanced interoperability)